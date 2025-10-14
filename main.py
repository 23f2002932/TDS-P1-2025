# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi [standard]",
#   "uvicorn",
#   "requests",
#   "python-dotenv",
#   "openai",
# ]
# ///

import requests
import os
import base64
import time
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# --- Setup and Configuration ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHARED_SECRET = os.getenv("secret")

if not all([GITHUB_TOKEN, OPENAI_API_KEY, SHARED_SECRET]):
    raise ValueError("Missing required environment variables: GITHUB_TOKEN, OPENAI_API_KEY, secret")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# --- Pydantic Models ---
class Attachment(BaseModel):
    name: str
    url: str

class TaskData(BaseModel):
    email: str
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[str]
    evaluation_url: str
    attachments: List[Attachment]

# --- Helper Functions ---
def validate_secret(secret: str) -> bool:
    """Validates the incoming secret against the one in the environment."""
    return secret == SHARED_SECRET

# --- GitHub API Functions ---
def create_github_repo(repo_name: str):
    """Creates a public GitHub repository with an MIT license."""
    payload = {"name": repo_name, "private": False, "auto_init": True, "license_template": "mit"}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if response.status_code == 201:
        print(f"Successfully created repo: {repo_name}")
    elif response.status_code == 422 and 'name already exists' in response.text:
        print(f"Repo {repo_name} already exists.")
    else:
        raise Exception(f"Failed to create repo: {response.status_code} {response.text}")

def enable_github_pages(repo_name: str, github_username: str):
    """Enables GitHub Pages for the specified repository."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    payload = {"source": {"branch": "main", "path": "/"}}
    url = f"https://api.github.com/repos/{github_username}/{repo_name}/pages"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        print(f"Successfully enabled GitHub Pages for {repo_name}")
    elif response.status_code == 409:
        print(f"GitHub Pages already enabled for {repo_name}")
    else:
        raise Exception(f"Failed to enable GitHub Pages: {response.status_code} {response.text}")

def get_file_content(repo_name: str, github_username: str, file_path: str) -> dict:
    """Fetches the content and SHA of a file from a GitHub repo."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{github_username}/{repo_name}/contents/{file_path}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content_data = response.json()
    decoded_content = base64.b64decode(content_data['content']).decode('utf-8')
    return {"content": decoded_content, "sha": content_data['sha']}

def push_files_to_repo(repo_name: str, github_username: str, files: List[Dict[str, str]]):
    """Pushes a list of files to a GitHub repository, creating or updating them."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    for file in files:
        file_name, file_content = file['name'], file['content']
        if not all([file_name, file_content]):
            continue
        
        url = f"https://api.github.com/repos/{github_username}/{repo_name}/contents/{file_name}"
        encoded_content = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
        payload = {"message": f"feat: Add or update {file_name}", "content": encoded_content, "branch": "main"}
        
        try:
            # Check if file exists to get its SHA for updating
            existing_file_data = get_file_content(repo_name, github_username, file_name)
            payload["sha"] = existing_file_data.get('sha')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:
                raise
            # File does not exist, which is fine for a create operation
        
        response = requests.put(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully pushed {file_name} to {repo_name}")

def get_latest_commit_sha(repo_name: str, github_username: str) -> str:
    """Gets the SHA of the latest commit on the main branch."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{github_username}/{repo_name}/commits/main"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['sha']

# --- LLM Functions ---
def _try_llm_models(messages: List[Dict[str, str]]):
    """Cycles through a list of LLM models, returning the first successful response."""
    models = ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"]
    for model in models:
        try:
            print(f"Attempting to generate content with model: {model}...")
            response = client.chat.completions.create(model=model, messages=messages)
            print(f"Successfully generated content with {model}.")
            return response
        except Exception as e:
            err_text = str(e).lower()
            if any(s in err_text for s in ['invalid_api_key', 'incorrect api key', '401']):
                raise HTTPException(status_code=401, detail=f"OpenAI API key error: {e}")
            elif any(s in err_text for s in ['model_not_found', 'does not exist', '404']):
                print(f"Model {model} not found or no access, trying next model.")
                continue
            else:
                print(f"An unexpected error occurred with model {model}: {e}")
    raise Exception("All LLM models failed to generate a response.")

def write_code_with_llm(brief: str) -> str:
    """Generates application code from a brief."""
    prompt = f"Generate a complete, single-file HTML application for this brief: \"{brief}\". Include all necessary HTML, CSS, and JavaScript. Respond with only the raw HTML code."
    messages = [{"role": "system", "content": "You are an expert web developer that creates single-file HTML applications."}, {"role": "user", "content": prompt}]
    response = _try_llm_models(messages)
    code = response.choices[0].message.content.strip()
    return code.removeprefix("```html").removesuffix("```").strip()

def modify_code_with_llm(brief: str, existing_code: str) -> str:
    """Generates modified code based on a new brief and existing code."""
    prompt = f"Modify the following HTML application based on this new request: \"{brief}\".\n\nExisting code:\n```html\n{existing_code}\n```\nReturn only the complete, raw, and updated HTML code."
    messages = [{"role": "system", "content": "You are an expert web developer that modifies existing code."}, {"role": "user", "content": prompt}]
    response = _try_llm_models(messages)
    code = response.choices[0].message.content.strip()
    return code.removeprefix("```html").removesuffix("```").strip()

def generate_readme_with_llm(brief: str) -> str:
    """Generates a professional README.md file from a brief."""
    prompt = f"Generate a professional README.md for a project based on this brief: \"{brief}\". Include a title, summary, setup instructions, and a license section (MIT License). Respond with only the raw markdown."
    messages = [{"role": "system", "content": "You are an expert technical writer."}, {"role": "user", "content": prompt}]
    response = _try_llm_models(messages)
    readme = response.choices[0].message.content.strip()
    return readme.removeprefix("```markdown").removesuffix("```").strip()

# --- Evaluation Submission ---
def submit_for_evaluation(data: TaskData, repo_url: str, commit_sha: str, pages_url: str):
    """Submits the final result to the evaluation URL with retry logic."""
    payload = {"email": data.email, "task": data.task, "round": data.round, "nonce": data.nonce,
               "repo_url": repo_url, "commit_sha": commit_sha, "pages_url": pages_url}
    for attempt in range(4):
        try:
            delay = 2 ** attempt
            response = requests.post(data.evaluation_url, json=payload, timeout=15)
            if response.status_code == 200:
                print("Successfully submitted for evaluation.")
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"status": "success", "response_body": response.text}
            else:
                print(f"Evaluation server returned status {response.status_code}. Retrying in {delay}s...")
        except requests.RequestException as e:
            print(f"Failed to connect to evaluation server: {e}. Retrying in {delay}s...")
        time.sleep(delay)
    raise Exception("Failed to submit for evaluation after several retries.")

# --- Task Processing Logic ---
def round1(data: TaskData):
    """Handles the entire Round 1 process."""
    repo_name = f"{data.task.replace(' ', '-')}-{data.nonce}"
    github_username = "23f2002932"
    try:
        create_github_repo(repo_name)
        new_html = write_code_with_llm(data.brief)
        new_readme = generate_readme_with_llm(data.brief)
        files_to_push = [{"name": "index.html", "content": new_html}, {"name": "README.md", "content": new_readme}]
        push_files_to_repo(repo_name, github_username, files_to_push)
        enable_github_pages(repo_name, github_username)
        print("Waiting 15 seconds for GitHub Pages and commit to propagate...")
        time.sleep(15)
        commit_sha = get_latest_commit_sha(repo_name, github_username)
        repo_url = f"https://github.com/{github_username}/{repo_name}"
        pages_url = f"https://{github_username}.github.io/{repo_name}/"
        evaluation_response = submit_for_evaluation(data, repo_url, commit_sha, pages_url)
        return {"message": "Round 1 completed and submitted successfully.", "repo_url": repo_url,
                "pages_url": pages_url, "commit_sha": commit_sha, "evaluation_response": evaluation_response}
    except Exception as e:
        print(f"An error occurred in round1: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during Round 1 processing: {str(e)}")

def round2(data: TaskData):
    """Handles the entire Round 2 process."""
    repo_name = f"{data.task.replace(' ', '-')}-{data.nonce}"
    github_username = "23f2002932"
    try:
        print(f"Fetching existing code from {repo_name}...")
        existing_file = get_file_content(repo_name, github_username, "index.html")
        modified_html = modify_code_with_llm(data.brief, existing_file["content"])
        updated_readme = generate_readme_with_llm(data.brief)
        files_to_push = [{"name": "index.html", "content": modified_html}, {"name": "README.md", "content": updated_readme}]
        push_files_to_repo(repo_name, github_username, files_to_push)
        print("Waiting 10 seconds for GitHub to update...")
        time.sleep(10)
        commit_sha = get_latest_commit_sha(repo_name, github_username)
        repo_url = f"https://github.com/{github_username}/{repo_name}"
        pages_url = f"https://{github_username}.github.io/{repo_name}/"
        evaluation_response = submit_for_evaluation(data, repo_url, commit_sha, pages_url)
        return {"message": "Round 2 completed and submitted successfully.", "repo_url": repo_url,
                "commit_sha": commit_sha, "evaluation_response": evaluation_response}
    except Exception as e:
        print(f"An error occurred in round2: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during Round 2 processing: {str(e)}")

# --- FastAPI Endpoint ---
@app.post("/handle_task")
def handle_task(data: TaskData):
    if not validate_secret(data.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    if data.round == 1:
        return round1(data)
    elif data.round == 2:
        return round2(data)
    else:
        raise HTTPException(status_code=400, detail="Invalid round number")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
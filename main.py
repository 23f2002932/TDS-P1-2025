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
import logging

# --- START: Final Logging Configuration for Hugging Face ---
# This simpler configuration logs all output directly to the console.
# The Hugging Face "Logs" tab will automatically capture and display this.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # No filename is specified, so output goes to the console
)
# --- END: Final Logging Configuration ---

# --- Setup and Configuration ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHARED_SECRET = os.getenv("secret")

if not all([GITHUB_TOKEN, OPENAI_API_KEY, SHARED_SECRET]):
    logging.critical("Missing required environment variables. Ensure GITHUB_TOKEN, OPENAI_API_KEY, and secret are set.")
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

# --- GitHub API Functions (with Logging) ---

def get_github_username() -> str:
    """Dynamically fetches the authenticated user's GitHub username."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get("https://api.github.com/user", headers=headers)
        response.raise_for_status()
        return response.json()['login']
    except requests.RequestException as e:
        logging.error(f"Failed to fetch GitHub username: {e}")
        raise Exception(f"Failed to fetch GitHub username: {e}")

def create_github_repo(repo_name: str):
    """Creates a public GitHub repository with an MIT license."""
    payload = {"name": repo_name, "private": False, "auto_init": True, "license_template": "mit"}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully created repo: {repo_name}")
    elif response.status_code == 422 and 'name already exists' in response.text:
        logging.warning(f"Repo {repo_name} already exists.")
    else:
        logging.error(f"Failed to create repo: {response.status_code} {response.text}")
        raise Exception(f"Failed to create repo: {response.status_code} {response.text}")

def enable_github_pages(repo_name: str, github_username: str):
    """Enables GitHub Pages for the specified repository."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    payload = {"source": {"branch": "main", "path": "/"}}
    url = f"https://api.github.com/repos/{github_username}/{repo_name}/pages"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully enabled GitHub Pages for {repo_name}")
    elif response.status_code == 409:
        logging.warning(f"GitHub Pages already enabled for {repo_name}")
    else:
        logging.error(f"Failed to enable GitHub Pages: {response.status_code} {response.text}")
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
        file_name, file_content = file.get('name'), file.get('content')
        if not all([file_name, file_content]):
            continue
        
        url = f"https://api.github.com/repos/{github_username}/{repo_name}/contents/{file_name}"
        encoded_content = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
        payload = {"message": f"feat: Add or update {file_name}", "content": encoded_content, "branch": "main"}
        
        try:
            existing_file_data = get_file_content(repo_name, github_username, file_name)
            payload["sha"] = existing_file_data.get('sha')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:
                raise
        
        response = requests.put(url, headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Successfully pushed {file_name} to {repo_name}")

def get_latest_commit_sha(repo_name: str, github_username: str) -> str:
    """Gets the SHA of the latest commit on the main branch."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{github_username}/{repo_name}/commits/main"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['sha']

def poll_github_pages_url(pages_url: str, timeout: int = 120, interval: int = 10):
    """Polls the GitHub Pages URL until it returns a 200 OK or times out."""
    start_time = time.time()
    logging.info(f"Polling GitHub Pages URL: {pages_url}")
    while time.time() - start_time < timeout:
        try:
            response = requests.get(pages_url, timeout=10)
            if response.status_code == 200:
                logging.info("GitHub Pages site is live.")
                return True
        except requests.RequestException:
            pass # Ignore connection errors while waiting
        logging.info(f"Site not yet live. Waiting {interval}s before next check...")
        time.sleep(interval)
    logging.error(f"GitHub Pages URL timed out after {timeout} seconds.")
    raise Exception(f"GitHub Pages URL did not become available within {timeout} seconds.")

# --- LLM Functions (with Logging) ---
def _try_llm_models(messages: List[Dict[str, str]], json_mode: bool = False):
    """Cycles through LLM models, returning the first successful response."""
    models = ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]
    for model in models:
        try:
            logging.info(f"Attempting to generate content with model: {model}...")
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
            logging.info(f"Successfully generated content with {model}.")
            return response
        except Exception as e:
            logging.error(f"Error with model {model}: {e}. Trying next model.")
    raise Exception("All LLM models failed to generate a response.")

def generate_code_with_llm(brief: str, attachments: List[Attachment]) -> List[Dict[str, str]]:
    """Generates application code from a brief and attachments."""
    attachment_details = "\n".join([f"- {att.name}" for att in attachments])
    prompt = (
        f"Generate a complete, single-file HTML application for this brief: \"{brief}\".\n"
        f"The following attachments are provided:\n{attachment_details}\n"
        "Your response must be a JSON object containing a list of files to create. "
        "Each file should be an object with 'name' (e.g., 'index.html') and 'content' keys. "
        "For this task, all code should be in a single 'index.html' file.\n"
        "Example format: {\"files\": [{\"name\": \"index.html\", \"content\": \"<!DOCTYPE html>...\"}]}"
    )
    messages = [{"role": "system", "content": "You are an expert web developer that creates single-file HTML applications and responds in JSON format."}, 
                {"role": "user", "content": prompt}]
    logging.info("Generating code with LLM...")
    response = _try_llm_models(messages, json_mode=True)
    content = response.choices[0].message.content.strip()
    return json.loads(content).get("files", [])

def modify_code_with_llm(brief: str, existing_files: List[Dict[str, str]], attachments: List[Attachment]) -> List[Dict[str, str]]:
    """Generates modified code based on a new brief, existing files, and attachments."""
    existing_files_str = json.dumps(existing_files, indent=2)
    attachment_details = "\n".join([f"- {att.name}" for att in attachments])
    prompt = (
        f"Modify the application based on this new request: \"{brief}\".\n"
        f"The following attachments are provided for this task:\n{attachment_details}\n"
        f"Here are the existing files:\n{existing_files_str}\n"
        "Your response must be a JSON object containing a list of all files for the updated project. "
        "Each file should be an object with 'name' and 'content' keys. You can modify existing files or add new ones."
    )
    messages = [{"role": "system", "content": "You are an expert web developer that modifies existing code and responds in JSON format."},
                {"role": "user", "content": prompt}]
    logging.info("Modifying code with LLM...")
    response = _try_llm_models(messages, json_mode=True)
    content = response.choices[0].message.content.strip()
    return json.loads(content).get("files", [])

def generate_readme_with_llm(brief: str) -> str:
    """Generates a professional README.md file from a brief."""
    prompt = f"Generate a professional README.md for a project based on this brief: \"{brief}\". Include a title, summary, setup instructions, and a license section (MIT License). Respond with only the raw markdown."
    messages = [{"role": "system", "content": "You are an expert technical writer."}, {"role": "user", "content": prompt}]
    logging.info("Generating README with LLM...")
    response = _try_llm_models(messages)
    readme = response.choices[0].message.content.strip()
    return readme.removeprefix("```markdown").removesuffix("```").strip()

# --- Evaluation Submission (with Logging) ---
def submit_for_evaluation(data: TaskData, repo_url: str, commit_sha: str, pages_url: str):
    """Submits the final result to the evaluation URL with retry logic."""
    payload = {"email": data.email, "task": data.task, "round": data.round, "nonce": data.nonce,
               "repo_url": repo_url, "commit_sha": commit_sha, "pages_url": pages_url}
    
    logging.info(f"Submitting to evaluation server. Payload: {json.dumps(payload, indent=2)}")
    
    for attempt in range(4):
        try:
            delay = 2 ** attempt
            response = requests.post(data.evaluation_url, json=payload, timeout=15)
            if response.status_code == 200:
                logging.info("Successfully submitted for evaluation.")
                return response.json() if response.content else {"status": "success"}
            else:
                logging.warning(f"Evaluation server returned status {response.status_code}. Retrying in {delay}s...")
        except requests.RequestException as e:
            logging.error(f"Failed to connect to evaluation server: {e}. Retrying in {delay}s...")
        if attempt < 3: time.sleep(delay)
    raise Exception("Failed to submit for evaluation after several retries.")

# --- Task Processing Logic ---
def round1(data: TaskData):
    """Handles the entire Round 1 process."""
    repo_name = f"{data.task.replace(' ', '-')}-{data.nonce}"
    github_username = get_github_username()
    try:
        create_github_repo(repo_name)
        files_to_push = generate_code_with_llm(data.brief, data.attachments)
        new_readme = generate_readme_with_llm(data.brief)
        files_to_push.append({"name": "README.md", "content": new_readme})
        
        push_files_to_repo(repo_name, github_username, files_to_push)
        enable_github_pages(repo_name, github_username)
        
        repo_url = f"https://github.com/{github_username}/{repo_name}"
        pages_url = f"https://{github_username}.github.io/{repo_name}/"
        
        poll_github_pages_url(pages_url)
        
        commit_sha = get_latest_commit_sha(repo_name, github_username)
        evaluation_response = submit_for_evaluation(data, repo_url, commit_sha, pages_url)
        return {"message": "Round 1 completed successfully.", "repo_url": repo_url, "pages_url": pages_url, "commit_sha": commit_sha, "evaluation_response": evaluation_response}
    except Exception as e:
        logging.error(f"An error occurred in round1: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during Round 1 processing: {str(e)}")

def round2(data: TaskData):
    """Handles the entire Round 2 process."""
    repo_name = f"{data.task.replace(' ', '-')}-{data.nonce}"
    github_username = get_github_username()
    try:
        logging.info(f"Fetching existing code from {repo_name} for Round 2...")
        existing_file = get_file_content(repo_name, github_username, "index.html")
        existing_files = [{"name": "index.html", "content": existing_file["content"]}]
        
        files_to_push = modify_code_with_llm(data.brief, existing_files, data.attachments)
        updated_readme = generate_readme_with_llm(data.brief)
        
        readme_found = False
        for file in files_to_push:
            if file['name'].lower() == 'readme.md':
                file['content'] = updated_readme
                readme_found = True
                break
        if not readme_found:
             files_to_push.append({"name": "README.md", "content": updated_readme})

        push_files_to_repo(repo_name, github_username, files_to_push)
        
        repo_url = f"https://github.com/{github_username}/{repo_name}"
        pages_url = f"https://{github_username}.github.io/{repo_name}/"
        
        poll_github_pages_url(pages_url)
        
        commit_sha = get_latest_commit_sha(repo_name, github_username)
        evaluation_response = submit_for_evaluation(data, repo_url, commit_sha, pages_url)
        return {"message": "Round 2 completed successfully.", "repo_url": repo_url, "commit_sha": commit_sha, "evaluation_response": evaluation_response}
    except Exception as e:
        logging.error(f"An error occurred in round2: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during Round 2 processing: {str(e)}")

# --- FastAPI Endpoint (with Logging) ---
@app.post("/handle_task")
def handle_task(data: TaskData):
    logging.info(f"--- NEW TASK RECEIVED (Round {data.round}) ---")
    logging.info(f"Received task data: {data.model_dump_json(indent=2)}")
    
    if not validate_secret(data.secret):
        logging.error("Invalid secret received. Rejecting request.")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    try:
        if data.round == 1:
            result = round1(data)
        elif data.round == 2:
            result = round2(data)
        else:
            logging.error(f"Invalid round number received: {data.round}")
            raise HTTPException(status_code=400, detail="Invalid round number")
        
        logging.info(f"--- TASK COMPLETED (Round {data.round}) ---")
        return result
        
    except Exception as e:
        logging.critical(f"A critical error occurred during task processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting up the FastAPI server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

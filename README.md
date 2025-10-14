# AI Developer Bot - TDS Project 1

This project is an autonomous agent capable of building, deploying, and updating single-page web applications based on a given brief. It leverages an LLM for code generation and the GitHub API for automated deployment to GitHub Pages. This project was developed as part of the TDS Project-1 curriculum.

![Robot working on a computer](https://t3.ftcdn.net/jpg/06/33/36/82/360_F_633368225_LwDAmhPhhsEwCLN8JT1cADEne6r848sr.jpg)

---

## Features

-   **🤖 Automated Build:** Receives a JSON request with an application brief, uses an LLM to generate the complete HTML, CSS, and JavaScript code.
-   **🚀 Automated Deployment:** Programmatically creates a new public GitHub repository, pushes the generated code, and enables GitHub Pages to make the application live.
-   **🔄 Automated Revision:** Handles a second request to fetch the existing code, modify it based on a new brief, and redeploy the updated application to the same GitHub Pages site.

---

## API Endpoint

The server listens for POST requests on a single endpoint to handle all tasks.

-   **URL:** `/handle_task`
-   **Method:** `POST`
-   **Body:** The request body must be a JSON object with the following structure:

```json
{
  "email": "student@example.com",
  "secret": "your_shared_secret",
  "task": "unique-task-name",
  "round": 1,
  "nonce": "unique-nonce-string",
  "brief": "A natural language description of the application to build or modify.",
  "checks": [],
  "evaluation_url": "https://example.com/evaluation_endpoint",
  "attachments": []
}
```

## Setup and Local Usage
To run this project on your local machine, follow these steps.

### Prerequisites
- Python 3.11 or higher
- A GitHub Personal Access Token with `repo` and `workflow` scopes
- An OpenAI API Key

### Installation
Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Set up environment variables
Create a `.env` file in the root of the project and add your secret keys:

```text
GITHUB_TOKEN="ghp_YourGitHubToken"
OPENAI_API_KEY="sk-YourOpenAIKey"
secret="YourChosenSecretPassword"
```

### Run the server

```bash
uvicorn main:app --reload
```

The server will start on `http://127.0.0.1:8000`.

## Deployment
This application is designed for deployment on cloud platforms like Vercel or Heroku.

- `Procfile` (for Heroku): `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment Variables: Ensure `GITHUB_TOKEN`, `OPENAI_API_KEY`, and `secret` are set in the deployment environment's configuration variables.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

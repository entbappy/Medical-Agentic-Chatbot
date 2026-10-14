# MedGuide AI — LangGraph + Pinecone + FastAPI

An educational medical assistant with retrieval from the bundled historical Gale Encyclopedia of Medicine (2002, Volume 1 A–B), optional Tavily fallback, image observations, speech input/output, and a vanilla JS web UI.

## Requirements
Python 3.11 or 3.12; OpenAI, Pinecone, and Tavily API keys; internet connection. This is a tutorial reference implementation, **not** a clinically validated diagnostic system or HIPAA-compliant deployment.

## Start
```bash
conda create -n project python=3.11 -y

conda activate project

pip install -r requirements.txt

add  .env

# Edit .env with real API keys

python ingest.py --pdf data/medical_book.pdf

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, then install and run the same commands.

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**. `/api/health` checks configuration. Run `ingest.py` once before chatting; it makes paid embedding calls and creates a Pinecone serverless index. The Pinecone index uses 1536 dimensions for `text-embedding-3-small`. If using a different embedding model, adjust the dimension and use a fresh index.

## Structure
- `app/config.py` — environment and Conda SSL certificate workaround
- `app/knowledge.py` — Pinecone vector retrieval
- `app/agent.py` — LangGraph retrieve → route → Tavily (when needed) → answer
- `app/main.py` — FastAPI, multimodal image observations, Whisper and TTS endpoints
- `ingest.py` — chunk and embed every readable PDF page
- `templates/index.html` — UI
- `static/style.css`, `static/app.js` — styles and browser controls
- `data/medical_book.pdf` — user-supplied reference PDF

## Troubleshooting
- **Connecting / checking**: run `uvicorn` and open `/api/health` first. Check browser console and backend terminal.
- **401**: if `API_TOKEN` is configured, enter it in the UI's API access section. Do not expose this token in a public browser deployment.
- **502**: inspect backend terminal; verify OpenAI/Pinecone keys, Pinecone service availability, embedding dimension and SSL certificate.
- **Camera/microphone**: browser permissions and localhost or HTTPS required.
- **No KB results**: run `python ingest.py` and wait for successful completion; ensure same index and API keys.
- **Tavily**: optional but needed for external fallback. Without its key, external search is disabled.

## Security and production limitations
This project is a functional tutorial scaffold, not a production medical device. Before deploying to real patients: replace shared API token with server-side identity/authentication; add database-backed encrypted conversations and retention/deletion controls; enforce request throttling and upload content verification; implement observability, tests, clinically reviewed emergency triage and evaluation; verify licensing to host/index the Gale PDF; obtain privacy/security/legal review; ensure current authoritative clinical references; and secure the deployment behind HTTPS. Do not send identifiable patient data to third-party APIs without appropriate consent and agreements.

The historical book may contain obsolete clinical advice; model instructions do not guarantee safety. Images can be described but cannot confirm or rule out disease. This software does not make diagnoses.

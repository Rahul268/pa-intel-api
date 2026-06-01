# PA Intelligence — Backend API

FastAPI wrapper around the full PA extraction pipeline.

## Status
- ✅ Code pushed to GitHub
- ⏳ Deploy to Render: see one-click button below

## One-Click Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Rahul268/pa-intel-api)

**After clicking:**
1. Sign in to Render (free account)
2. Render auto-reads `render.yaml` — everything is pre-configured
3. Under **Environment Variables** → add `GROQ_API_KEY` = your Groq key
4. Click **Deploy Web Service**
5. Your API URL: `https://pa-intel-api.onrender.com`

## Pipeline
PDF Upload → `parse_pdf` → `chunk_pdf_pages` → `build_index` → `retrieve_evidence` → `reduce_evidence_context` → `extract_row` → `validate_step_counts` → `normalize_row` → `calculate_access_score`

## Endpoints
- `GET /health` — liveness probe
- `POST /api/extract` — upload PDF + brand, returns 12 parameters + access score

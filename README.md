# AI Resume Matcher (Streamlit)

A Streamlit app that loads candidate and job data from Airtable, filters candidates, and uses Anthropic Claude to score and rank candidates based on job requirements.

## Features
- Fast parallel Airtable fetch with caching
- Flexible filtering by role, function, industry, CTC, experience, and keywords
- Optional custom AI scoring prompt
- Ranked results with summaries and charts

## Tech Stack
- Streamlit
- Airtable API
- Anthropic Claude
- Plotly + Pandas

## Setup
1. Create a virtual environment (optional but recommended)
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set environment variables

Option A: `.env` (local dev)
```bash
cp .env.example .env
# edit .env with your keys
```
`python-dotenv` is used to load `.env` automatically.

Option B: Streamlit secrets (recommended)
Create `.streamlit/secrets.toml`:
```toml
AIRTABLE_API_KEY = "your_airtable_key"
ANTHROPIC_API_KEY = "your_anthropic_key"
```

## Run
```bash
python3 -m streamlit run v8.py
```

## Configuration
Use the sidebar inputs to set:
- Jobs Table ID
- Jobs View ID

Candidate data is loaded from the Airtable base configured in `v8.py` (`BASE_URL`).

## Notes
- Rotate any keys that were previously committed to the repo.
- `.streamlit/secrets.toml` is gitignored by default.

## License
Add a license of your choice (MIT, Apache-2.0, etc.).

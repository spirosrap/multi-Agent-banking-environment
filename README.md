# Multi-Agent Banking System

This project implements a multi-agent banking assistant using Google ADK, A2A, and Toolbox SQL tools. A manager agent routes requests to deposit and loan specialists, while the loan agent orchestrates a structured approval workflow that uses policy and customer profile PDFs stored in GCS.

## Architecture

- Manager agent: routes requests and answers basic general banking questions.
- Deposit agent: account lists, per-account balances, transactions, and minimum balance checks.
- Loan agent: loan lookups plus full loan approval orchestration with policy and profile evaluation.

## Requirements

- Python 3.10+
- MySQL (local or Cloud SQL) with a `bank` database
- Google Cloud credentials for GCS access
- Google ADK CLI (`adk`) on your PATH
- Toolbox server (binary or equivalent)

## Setup

1) Create and activate a Python environment.

Conda:
```bash
conda create -n gcp-agenticai-c4 python=3.10
conda activate gcp-agenticai-c4
```

venv:
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies.
```bash
pip install -r starter/requirements.txt
```

3) Configure environment variables.
```bash
cp starter/.env-sample starter/.env
set -a
source starter/.env
set +a
```

Required values in `starter/.env`:
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `TOOLBOX_URL` (default `http://127.0.0.1:5001`)
- `POLICY_PDF_URI`, `CUSTOMER_PROFILE_PDF_URI`

4) Create and seed the MySQL database.
```bash
mysql -u <user> -p -e "CREATE DATABASE bank;"
mysql -u <user> -p bank < starter/docs/deposit.sql
mysql -u <user> -p bank < starter/docs/loan.sql
```

5) Upload the policy PDFs to GCS and update the URIs.
```bash
gsutil mb gs://<bucket_name>
gsutil cp starter/docs/loan-policy.pdf gs://<bucket_name>/loan-policy.pdf
gsutil cp starter/docs/loan-customer-info.pdf gs://<bucket_name>/loan-customer-info.pdf
```

## Run Locally

1) Start the Toolbox server (example uses the bundled binary in repo root).
```bash
./toolbox --tools-file starter/tools.yaml --port 5001
```

2) Start the ADK web UI with A2A enabled.
```bash
adk web --a2a --host 127.0.0.1 --port 8000 starter
```

3) Open the UI:
```
http://127.0.0.1:8000/dev-ui/
```

## Testing

Run the functional test script and generate evidence artifacts:
```bash
python testing/bin/a2a.py --in testing/test_scenarios.csv --format csv json txt --out test_results
```

Evidence files:
- `test_results.csv`, `test_results.json`, `test_results.txt`
- `testing/loan-approval.png`, `testing/loan-rejection.png`

## Project Structure

- `starter/` - all agent implementations and prompts
- `starter/docs/` - SQL seeds and PDF documents
- `testing/` - test scenarios and evidence screenshots
- `REPORT.md` - report and risk analysis

## Notes

- The deposit agent never reveals total balances across all accounts.
- Loan approvals require both the policy criteria and the equity check to pass.

# Multi-Agent Banking System Report

## Architecture Overview

This system is built around three agents that collaborate over A2A:

- **Manager agent (`starter/manager/agent.py`, `starter/manager/agent.json`)**: routes customer requests to the correct specialist (deposit or loan) and answers basic general banking questions without routing.
- **Deposit agent (`starter/deposit/agent.py`, `starter/deposit/agent.json`)**: handles deposit account listings, per‑account balances, and transactions using Toolbox SQL tools. It also performs equity checks via `check_minimum_balance` without revealing total deposits.
- **Loan agent (`starter/loan/loan.py`, `starter/loan/agent.json`)**: handles loan lookups and orchestrates the full loan approval workflow. It uses Toolbox SQL for loan balances and A2A to call the deposit agent for equity checks.

Interaction flow:
- User → Manager agent.
- Manager routes to Deposit or Loan agent via A2A, depending on the request.
- Loan agent uses A2A to call the Deposit agent for equity checks (no direct code imports between agents).

## Loan Approval Orchestration (Detailed)

The loan approval workflow is an orchestrated pipeline that combines sequential and parallel steps and uses structured state outputs:

1. **Loan request parsing**: `loan_request_agent` extracts `customer_id`, `loan_type`, and `loan_amount` into `loan_request`.
2. **Parallel checks**:
   - **Outstanding balance**: `outstanding_balance_agent` calls the loan DB tool and stores `outstanding_summary`.
   - **Policy evaluation**: `policy_pdf_loader` loads the policy PDF from GCS, then `policy_criteria_agent` extracts `debt_to_equity_ratio` and `min_customer_rating` into `policy_criteria`.
   - **Customer profile evaluation**: `customer_profile_loader` loads the customer profile PDF from GCS, then `customer_profile_agent` extracts `customer_rating` into `customer_profile`.
3. **Minimum equity calculation**: `minimum_equity_agent` computes required equity from the policy ratio and total debt, writing `required_equity`.
4. **Equity check via A2A**: `equity_check_agent` calls the deposit agent with `check_minimum_balance` and stores `equity_check`.
5. **Final decision**: `approval_report_agent` uses all state values and returns a structured `approval_decision` that is customer‑safe and does not reveal policy thresholds or ratings.

All sub‑agents use `output_schema` and `output_key` to keep state structured and machine‑readable.

## Test Results Evaluation

Evidence files:
- `test_results.csv`, `test_results.json`, `test_results.txt`
- `testing/loan-approval.png`, `testing/loan-rejection.png`

What the tests show:
- **Strength**: Guardrails work. The deposit agent refuses to disclose total deposit balances and rejects unauthorized actions (adding funds). The loan workflow produces both approve and decline decisions with final state visible in the UI.
- **Area for improvement**: The test log shows at least one tool invocation error (`INVALID_ARGUMENT`) during a loan prompt in the manager flow. This indicates the tool/mime-type handling could be hardened or the manager could better detect loan‑approval intents and route to the loan agent without triggering unsupported tool responses.

## Risks and Mitigations

1. **LLM hallucinations or inconsistent decisions**
   - *Risk*: Incorrect policy interpretation or inconsistent outcomes.
   - *Mitigation*: Keep critical calculations (equity) deterministic, validate required fields, and gate approvals with rule‑based checks; add automated regression tests.
2. **Removing humans from the decision loop**
   - *Risk*: Approvals/declines happen without oversight, leading to policy or compliance risk.
   - *Mitigation*: Require human review for high‑value loans or borderline cases; log all decisions for audit and review.
3. **Sensitive data exposure**
   - *Risk*: Agents could reveal totals or policy thresholds.
   - *Mitigation*: Enforce tool‑level constraints, prompt guardrails, and response filters; add redaction and PII detection for outbound responses.

## Future Improvements

1. Add policy versioning and automated regression tests so changes to PDFs or prompts are validated before release.
2. Integrate external credit‑risk inputs (credit bureau/KYC) and add a human‑approval queue for high‑risk cases.
3. Improve observability: structured logs, decision metrics, and alerts for tool errors or policy mismatches.
4. Strengthen authentication and authorization for A2A requests to ensure only approved agents can invoke sensitive tools.

import io
import os
from typing import AsyncGenerator, Optional, List

from pydantic import BaseModel, Field
from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent, BaseAgent, InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.events import Event, EventActions
from google.cloud import storage
from google.genai.types import Content, Part
from pdfminer.high_level import extract_text

from toolbox_core import ToolboxSyncClient

MODEL_NAME = "gemini-2.5-flash"
DEFAULT_POLICY_PDF_URI = "gs://udacity-spiros-1/loan-policy.pdf"
DEFAULT_CUSTOMER_PROFILE_PDF_URI = "gs://udacity-spiros-1/loan-customer-info.pdf"


def load_instructions(prompt_file: str) -> str:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  instruction_file_path = os.path.join(script_dir, prompt_file)
  with open(instruction_file_path, "r") as f:
    return f.read()


def build_prompt(prompt_file: str) -> str:
  base_prompt = load_instructions("user-profile-base-prompt.txt")
  prompt = load_instructions(prompt_file)
  if base_prompt and prompt:
    return f"{base_prompt}\n\n{prompt}"
  return base_prompt or prompt


def parse_gcs_uri(uri: str) -> tuple[str, str]:
  if not uri.startswith("gs://"):
    raise ValueError("Expected a gs:// bucket URI.")
  bucket_and_blob = uri[len("gs://"):]
  bucket_name, _, blob_name = bucket_and_blob.partition("/")
  if not bucket_name or not blob_name:
    raise ValueError("GCS URI must include bucket and object name.")
  return bucket_name, blob_name


def load_pdf_text(uri: str) -> str:
  bucket_name, blob_name = parse_gcs_uri(uri)
  client = storage.Client()
  blob = client.bucket(bucket_name).blob(blob_name)
  pdf_bytes = blob.download_as_bytes()
  with io.BytesIO(pdf_bytes) as pdf_stream:
    return extract_text(pdf_stream) or ""


def to_float(value) -> Optional[float]:
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


class LoanRequestOutput(BaseModel):
  customer_id: int = Field(description="Customer ID for the loan request.")
  loan_type: str = Field(description="Requested loan type.")
  loan_amount: float = Field(description="Requested loan amount.")


class OutstandingLoan(BaseModel):
  loan_id: int = Field(description="Loan ID.")
  loan_type: str = Field(description="Type of the outstanding loan.")
  outstanding_balance: float = Field(description="Outstanding balance for the loan.")


class OutstandingBalanceOutput(BaseModel):
  loans: List[OutstandingLoan] = Field(description="Outstanding loans for the customer.")
  total_outstanding: float = Field(description="Sum of outstanding balances.")


class PolicyCriteriaOutput(BaseModel):
  debt_to_equity_ratio: float = Field(description="Required debt-to-equity ratio from policy.")
  min_customer_rating: str = Field(description="Minimum customer rating required by policy.")
  policy_summary: Optional[str] = Field(default=None, description="Short policy summary used.")


class CustomerProfileOutput(BaseModel):
  customer_rating: str = Field(description="Customer rating from the profile.")
  profile_summary: Optional[str] = Field(default=None, description="Short customer profile summary.")


class EquityCheckOutput(BaseModel):
  meets_minimum: bool = Field(description="Whether deposits meet the required equity.")
  reason: str = Field(description="Short rationale for the equity check.")


class ApprovalDecisionOutput(BaseModel):
  decision: str = Field(description="Approve or Decline.")
  reasons: List[str] = Field(description="Reasons for the decision.")
  next_step: Optional[str] = Field(default=None, description="Suggested next step if declined.")


class PdfLoaderAgent(BaseAgent):
  source_env_var: str
  default_uri: str
  output_key: str

  def __init__(self, name: str, source_env_var: str, default_uri: str, output_key: str):
    super().__init__(
      name=name,
      source_env_var=source_env_var,
      default_uri=default_uri,
      output_key=output_key,
    )

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    uri = os.environ.get(self.source_env_var, self.default_uri)
    error = None
    try:
      text = load_pdf_text(uri)
    except Exception as exc:
      text = ""
      error = str(exc)

    payload = {"uri": uri, "text": text}
    if error:
      payload["error"] = error

    summary = "Loaded PDF content." if not error else "Failed to load PDF content."
    event = Event(
      invocation_id=ctx.invocation_id,
      author=self.name,
      branch=ctx.branch,
      content=Content(parts=[Part(text=summary)]),
      actions=EventActions(state_delta={self.output_key: payload}),
    )
    yield event


class MinimumEquityAgent(BaseAgent):
  output_key: str = "required_equity"

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    state = ctx.session.state
    loan_request = state.get("loan_request", {})
    outstanding_summary = state.get("outstanding_summary", {})
    policy_criteria = state.get("policy_criteria", {})

    loan_amount = to_float(loan_request.get("loan_amount"))
    total_outstanding = to_float(outstanding_summary.get("total_outstanding")) or 0.0
    ratio = to_float(policy_criteria.get("debt_to_equity_ratio"))

    total_debt = None
    required_equity = None
    error = None

    if loan_amount is None or ratio in (None, 0.0):
      error = "Missing loan amount or debt-to-equity ratio."
    else:
      total_debt = loan_amount + total_outstanding
      required_equity = round(total_debt / ratio, 2)

    payload = {
      "loan_amount": loan_amount,
      "total_outstanding": total_outstanding,
      "debt_to_equity_ratio": ratio,
      "total_debt": total_debt,
      "required_equity": required_equity,
    }
    if error:
      payload["error"] = error

    event = Event(
      invocation_id=ctx.invocation_id,
      author=self.name,
      branch=ctx.branch,
      content=Content(parts=[Part(text="Computed minimum required equity.")]),
      actions=EventActions(state_delta={self.output_key: payload}),
    )
    yield event


toolbox_url = os.environ.get(
  "LOAN_TOOLBOX_URL",
  os.environ.get("TOOLBOX_URL", "http://127.0.0.1:5001"),
)
db_client = ToolboxSyncClient(toolbox_url)
loan_tools = [
  db_client.load_tool("get_outstanding_balances_by_customer"),
]

deposit_agent_url = os.environ.get(
  "DEPOSIT_AGENT_URL",
  f"http://localhost:8000/a2a/deposit{AGENT_CARD_WELL_KNOWN_PATH}",
)
deposit_agent = RemoteA2aAgent(
  name="deposit_agent",
  agent_card=deposit_agent_url,
  description="Deposit agent for equity checks.",
)

loan_request_agent = LlmAgent(
  name="loan_request_agent",
  description="Collects loan request details.",
  model=MODEL_NAME,
  instruction=build_prompt("loan-request-prompt.txt"),
  output_schema=LoanRequestOutput,
  output_key="loan_request",
)

outstanding_balance_agent = LlmAgent(
  name="outstanding_balance_agent",
  description="Fetches outstanding loan balances for a customer.",
  model=MODEL_NAME,
  instruction=build_prompt("outstanding-balance-prompt.txt"),
  tools=loan_tools,
  output_schema=OutstandingBalanceOutput,
  output_key="outstanding_summary",
)

policy_pdf_loader = PdfLoaderAgent(
  name="policy_pdf_loader",
  source_env_var="POLICY_PDF_URI",
  default_uri=DEFAULT_POLICY_PDF_URI,
  output_key="policy_document",
)

policy_criteria_agent = LlmAgent(
  name="policy_criteria_agent",
  description="Extracts policy criteria from the loan policy document.",
  model=MODEL_NAME,
  instruction=build_prompt("policy-criteria-prompt.txt"),
  output_schema=PolicyCriteriaOutput,
  output_key="policy_criteria",
)

policy_evaluation_agent = SequentialAgent(
  name="policy_evaluation_agent",
  description="Loads and evaluates loan policy criteria.",
  sub_agents=[policy_pdf_loader, policy_criteria_agent],
)

customer_profile_loader = PdfLoaderAgent(
  name="customer_profile_loader",
  source_env_var="CUSTOMER_PROFILE_PDF_URI",
  default_uri=DEFAULT_CUSTOMER_PROFILE_PDF_URI,
  output_key="customer_profile_document",
)

customer_profile_agent = LlmAgent(
  name="customer_profile_agent",
  description="Evaluates the customer profile from the document.",
  model=MODEL_NAME,
  instruction=build_prompt("customer-profile-prompt.txt"),
  output_schema=CustomerProfileOutput,
  output_key="customer_profile",
)

customer_profile_pipeline = SequentialAgent(
  name="customer_profile_pipeline",
  description="Loads and evaluates customer profile information.",
  sub_agents=[customer_profile_loader, customer_profile_agent],
)

minimum_equity_agent = MinimumEquityAgent(name="minimum_equity_agent")

equity_check_agent = LlmAgent(
  name="equity_check_agent",
  description="Checks deposit equity using the deposit agent.",
  model=MODEL_NAME,
  instruction=build_prompt("check-equity-prompt.txt"),
  sub_agents=[deposit_agent],
  output_schema=EquityCheckOutput,
  output_key="equity_check",
)

approval_report_agent = LlmAgent(
  name="approval_report_agent",
  description="Provides the final loan approval decision.",
  model=MODEL_NAME,
  instruction=build_prompt("approval-report-prompt.txt"),
  output_schema=ApprovalDecisionOutput,
  output_key="approval_decision",
)

parallel_checks_agent = ParallelAgent(
  name="parallel_checks_agent",
  description="Runs balance, policy, and profile checks in parallel.",
  sub_agents=[
    outstanding_balance_agent,
    policy_evaluation_agent,
    customer_profile_pipeline,
  ],
)

loan_approval_agent = SequentialAgent(
  name="loan_approval_agent",
  description="Runs the loan approval workflow end-to-end.",
  sub_agents=[
    loan_request_agent,
    parallel_checks_agent,
    minimum_equity_agent,
    equity_check_agent,
    approval_report_agent,
  ],
)

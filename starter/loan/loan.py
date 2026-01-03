import os
from typing import AsyncGenerator

from google.adk.agents import SequentialAgent, LlmAgent, BaseAgent, InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part

from toolbox_core import ToolboxSyncClient

MODEL_NAME = "gemini-2.5-flash"


def load_instructions( prompt_file: str ) -> str:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  instruction_file_path = os.path.join( script_dir, prompt_file )
  with open(instruction_file_path, "r") as f:
    return f.read()


def build_prompt( prompt_file: str ) -> str:
  base_prompt = load_instructions( "user-profile-base-prompt.txt" )
  prompt = load_instructions( prompt_file )
  if base_prompt and prompt:
    return f"{base_prompt}\n\n{prompt}"
  return base_prompt or prompt


toolbox_url = os.environ.get(
  "LOAN_TOOLBOX_URL",
  os.environ.get("TOOLBOX_URL", "http://127.0.0.1:5001"),
)
db_client = ToolboxSyncClient( toolbox_url )
loan_tools = [
  db_client.load_tool( "get_outstanding_balances_by_customer" ),
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
  instruction=build_prompt( "loan-request-prompt.txt" ),
)

outstanding_balance_agent = LlmAgent(
  name="outstanding_balance_agent",
  description="Fetches outstanding loan balances for a customer.",
  model=MODEL_NAME,
  instruction=build_prompt( "outstanding-balance-prompt.txt" ),
  tools=loan_tools,
)

equity_check_agent = LlmAgent(
  name="equity_check_agent",
  description="Checks deposit equity using the deposit agent.",
  model=MODEL_NAME,
  instruction=build_prompt( "check-equity-prompt.txt" ),
  sub_agents=[deposit_agent],
)

approval_report_agent = LlmAgent(
  name="approval_report_agent",
  description="Provides the final loan approval decision.",
  model=MODEL_NAME,
  instruction=build_prompt( "approval-report-prompt.txt" ),
)


class TotalValueAgent(BaseAgent):
  output_key: str = "loan_totals"

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    summary = "Reviewed the loan request and outstanding balances for approval."
    event = Event(
      invocation_id=ctx.invocation_id,
      author=self.name,
      branch=ctx.branch,
      content=Content( parts=[Part( text=summary )] ),
      actions=EventActions( state_delta={self.output_key: {"status": "reviewed"}} ),
    )
    yield event


total_value_agent = TotalValueAgent( name="total_value_agent" )

loan_approval_agent = SequentialAgent(
  name="loan_approval_agent",
  description="Runs the loan approval workflow end-to-end.",
  sub_agents=[
    loan_request_agent,
    outstanding_balance_agent,
    total_value_agent,
    equity_check_agent,
    approval_report_agent,
  ],
)

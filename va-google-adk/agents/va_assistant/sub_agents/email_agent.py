"""Email domain expert — sends invoices and quotes by email."""

from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

from ....shared.schema import AssistantResponse
from ....shared.tools.emails import send_invoice_by_email, send_quote_by_email
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "email_agent.txt").read_text()

email_agent = Agent(
    model="gemini-2.5-flash",
    name="email_agent",
    description="Sends invoices and quotes by email to customers. Drafts professional Danish email subjects and bodies.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[send_invoice_by_email, send_quote_by_email, report_out_of_domain],
    generate_content_config=THINKING_CONFIG,
)

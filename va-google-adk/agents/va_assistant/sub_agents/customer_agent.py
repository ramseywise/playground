"""Customer domain expert."""

from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

from ....shared.schema import AssistantResponse
from ....shared.tools.customers import create_customer, edit_customer, list_customers
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "customer_agent.txt").read_text()

customer_agent = Agent(
    model="gemini-2.5-flash",
    name="customer_agent",
    description="Handles customers and contacts: create, view, list, and edit. Knows CVR, address, and contact persons.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[list_customers, create_customer, edit_customer, report_out_of_domain],
    generate_content_config=THINKING_CONFIG,
)

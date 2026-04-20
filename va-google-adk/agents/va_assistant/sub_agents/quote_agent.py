"""Quote domain expert."""

from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

from ....shared.schema import AssistantResponse
from ....shared.tools.customers import list_customers
from ....shared.tools.products import list_products
from ....shared.tools.quotes import (
    create_invoice_from_quote,
    create_quote,
    list_quotes,
)
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "quote_agent.txt").read_text()

quote_agent = Agent(
    model="gemini-2.5-flash",
    name="quote_agent",
    description=(
        "Handles quotes: create, view, list, and convert to invoice. "
        "States: open, accepted, declined, invoiced, closed."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        list_quotes,
        create_quote,
        create_invoice_from_quote,
        list_customers,
        list_products,
        report_out_of_domain,
    ],
    generate_content_config=THINKING_CONFIG,
)

"""Product domain expert."""

from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

from ....shared.schema import AssistantResponse
from ....shared.tools.products import create_product, edit_product, list_products
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "product_agent.txt").read_text()

product_agent = Agent(
    model="gemini-2.5-flash",
    name="product_agent",
    description="Handles products and services: create, view, list, and edit prices. Products are used in invoice and quote line items.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[list_products, create_product, edit_product, report_out_of_domain],
    generate_content_config=THINKING_CONFIG,
)

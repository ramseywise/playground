"""Invitation domain expert."""

from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

from ....shared.schema import AssistantResponse
from ....shared.tools.invitations import invite_user
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "invitation_agent.txt").read_text()

invitation_agent = Agent(
    model="gemini-2.5-flash",
    name="invitation_agent",
    description="Invites new users to the Billy organization as collaborators by email.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[invite_user, report_out_of_domain],
    generate_content_config=THINKING_CONFIG,
)

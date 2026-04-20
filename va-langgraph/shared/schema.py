"""Shared output schema for the VA assistant.

Both gateway and agent layers import from here.  The frontend expects every
agent response to conform to AssistantResponse — same fields regardless of
which backend (ADK or LangGraph) is serving the request.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class NavButton(BaseModel):
    """A deep-link button rendered in the Billy app UI."""

    label: str = Field(description="Button label shown to the user.")
    route: str = Field(description="Billy app route, e.g. '/invoices' or '/invoices/inv_001'.")
    id: Optional[str] = Field(default=None, description="Entity ID for the button target.")
    document_type: Optional[str] = Field(
        default=None,
        description="Entity type, e.g. 'invoice', 'customer', 'quote'.",
    )


class Source(BaseModel):
    """A support documentation link surfaced alongside a support answer."""

    title: str = Field(description="Article or page title.")
    url: str = Field(description="Full URL to the support article.")


class FormConfig(BaseModel):
    """Triggers an inline creation form in the UI."""

    type: Literal["create_customer", "create_product", "create_invoice", "create_quote"] = Field(
        description="The form type to render."
    )
    defaults: Optional[dict] = Field(
        default=None,
        description="Pre-filled field values for the form, e.g. {'contactId': 'cus_001'}.",
    )


class EmailFormConfig(BaseModel):
    """Triggers an editable email composition form in the UI."""

    to: Optional[str] = Field(default=None, description="Recipient email address.")
    subject: Optional[str] = Field(default=None, description="Pre-filled email subject.")
    body: Optional[str] = Field(default=None, description="Pre-filled email body.")


class AssistantResponse(BaseModel):
    """The structured response every VA agent turn must produce.

    The ``message`` field is always required.  All other fields are optional
    and default to empty / False so the frontend can safely read them without
    null checks.
    """

    message: str = Field(
        description=(
            "Main response content as markdown.  This is what the user reads. "
            "Never include JSON or schema markers in this field."
        )
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="2-4 suggested follow-up actions shown as clickable chips.",
    )
    nav_buttons: list[NavButton] = Field(
        default_factory=list,
        description="Deep-link buttons that open the relevant page in the Billy app.",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Support article links, populated by the support agent.",
    )
    table_type: Optional[Literal["invoices", "customers", "products", "quotes"]] = Field(
        default=None,
        description="Set when the message contains a listing table of that entity type.",
    )
    form: Optional[FormConfig] = Field(
        default=None,
        description="Set to trigger an inline creation form in the UI.",
    )
    email_form: Optional[EmailFormConfig] = Field(
        default=None,
        description="Set to trigger an editable email composition form.",
    )
    confirm: bool = Field(
        default=False,
        description="Set to True to show Confirm/Discard buttons for pending edit operations.",
    )
    contact_support: bool = Field(
        default=False,
        description="Set to True to surface a Contact Customer Service button.",
    )

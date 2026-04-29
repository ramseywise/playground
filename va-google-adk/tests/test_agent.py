"""Smoke tests for the ADK agent structure — no API calls."""

from __future__ import annotations

import pytest
from google.adk.agents import Agent

from agent import root_agent as va_assistant
from sub_agents.accounting_agent import accounting_agent
from sub_agents.banking_agent import banking_agent
from sub_agents.customer_agent import customer_agent
from sub_agents.email_agent import email_agent
from sub_agents.expense_agent import expense_agent
from sub_agents.insights_agent import insights_agent
from sub_agents.invitation_agent import invitation_agent
from sub_agents.invoice_agent import invoice_agent
from sub_agents.product_agent import product_agent
from sub_agents.quote_agent import quote_agent
from sub_agents.support_agent import support_agent

_ALL_SUB_AGENTS = [
    accounting_agent,
    banking_agent,
    customer_agent,
    email_agent,
    expense_agent,
    insights_agent,
    invitation_agent,
    invoice_agent,
    product_agent,
    quote_agent,
    support_agent,
]


class TestAgentStructure:
    def test_root_agent_is_agent_instance(self):
        assert isinstance(va_assistant, Agent)

    def test_root_agent_has_all_sub_agents(self):
        sub_names = {a.name for a in va_assistant.sub_agents}
        expected = {a.name for a in _ALL_SUB_AGENTS}
        assert expected == sub_names

    @pytest.mark.parametrize("agent", _ALL_SUB_AGENTS, ids=lambda a: a.name)
    def test_sub_agent_is_agent_instance(self, agent):
        assert isinstance(agent, Agent)

    @pytest.mark.parametrize("agent", _ALL_SUB_AGENTS, ids=lambda a: a.name)
    def test_sub_agent_has_description(self, agent):
        assert agent.description and len(agent.description) > 10

    @pytest.mark.parametrize("agent", _ALL_SUB_AGENTS, ids=lambda a: a.name)
    def test_sub_agent_uses_gemini(self, agent):
        assert "gemini" in agent.model.lower()

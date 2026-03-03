#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base class for KORAL Stage II agents."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from stage_II.llm.openai_client import OpenAIChatClient, LLMResponse
from stage_II.utils.json_utils import extract_json_object


@dataclass
class AgentResult:
    """Standard result from any agent."""
    agent_name: str
    output: Dict[str, Any]        # parsed structured output
    raw_text: str = ""            # raw LLM response text (if applicable)
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0


class Agent(ABC):
    """Base class for all KORAL agents.

    Each agent encapsulates a single responsibility:
    - Receives structured input
    - Optionally calls the LLM with a specialized prompt
    - Returns a structured AgentResult

    Some agents (like the Evaluator) may not need LLM calls at all
    and can operate purely with deterministic logic.
    """

    def __init__(self, llm: Optional[OpenAIChatClient] = None, temperature: float = 0.2):
        self.llm = llm
        self.temperature = temperature

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """Execute the agent's task. Each subclass defines its own kwargs."""
        ...

    def _call_llm(self, system: str, user: str, max_tokens: int = 900,
                  seed: Optional[int] = None) -> Dict[str, Any]:
        """Helper: call LLM and parse JSON response.

        Returns the parsed dict, or a dict with 'parse_error' if parsing fails.
        """
        if self.llm is None:
            raise RuntimeError(f"Agent '{self.name}' requires an LLM client but none was provided.")

        resp: LLMResponse = self.llm.chat(
            system=system,
            user=user,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        parsed = extract_json_object(resp.text)
        if parsed is None:
            parsed = {"parse_error": True, "raw_text": resp.text}

        return parsed

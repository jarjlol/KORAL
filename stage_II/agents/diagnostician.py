#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostician agent — synthesizes telemetry + literature into a unified Diagnosis.

This is the CENTRAL reasoning agent. It creates a shared context (Diagnosis)
that ALL downstream task agents consume, ensuring cross-task consistency.

What it does:
  1. Receives TelemetrySummary (from Telemetry Analyst) + Literature evidence.
  2. Sends a focused prompt asking for causal reasoning: WHY is the drive
     in its current state? What failure modes are active?
  3. Returns a structured Diagnosis with health state, failure modes,
     contributing factors, and confidence assessment.

Why this helps:
  - In the baseline, each task (predictive/descriptive/prescriptive/what-if)
    independently re-interprets the same data, leading to contradictions.
  - The Diagnosis is a SHARED artifact — all 4 task agents read it.
  - This directly targets Cross-Task Consistency (CTC) improvement.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from stage_II.agents.base import Agent, AgentResult


DIAGNOSTICIAN_SYSTEM_PROMPT = """You are the Diagnostician agent in the KORAL SSD analysis pipeline.
You receive a telemetry health summary and literature evidence.
Your job is to synthesize a unified DIAGNOSIS explaining the drive's condition.
This diagnosis will be shared with downstream agents for predictive, descriptive,
prescriptive, and what-if analysis — so be precise and cite evidence.
Return ONLY valid JSON, no markdown fences."""

DIAGNOSTICIAN_USER_PROMPT_TEMPLATE = """Synthesize a diagnosis from these inputs.

Telemetry Summary (from Telemetry Analyst):
{telemetry_summary}

Literature Evidence:
{literature}

Available reference IDs you may cite: {available_refs}

Produce JSON with:
{{
  "health_state": <"healthy"|"watch"|"degrading"|"critical">,
  "risk_score": <0.0-1.0>,
  "primary_failure_mode": <string describing the main failure mechanism, or null>,
  "contributing_factors": [
    {{
      "factor": <text>,
      "evidence": [<ref_id>, ...],
      "severity": <"low"|"medium"|"high">
    }}
  ],
  "estimated_ttf_days": <number|null>,
  "confidence": <0.0-1.0>,
  "uncertainties": [<text>, ...],
  "diagnosis_summary": <2-3 sentence summary of drive condition>
}}

Rules:
- Every contributing_factor MUST cite at least one evidence ref.
- Use IR refs (e.g., "AF_r_187") for device-specific findings.
- Use Literature refs (e.g., "LIT_1") for mechanism explanations.
- Set confidence lower if data quality flags were raised.
- estimated_ttf_days should be null if insufficient evidence.
"""


class Diagnostician(Agent):
    """Synthesizes telemetry analysis + literature into a unified Diagnosis.

    Pipeline position: MIDDLE agent, runs after Telemetry Analyst + Literature retrieval.
    Input: TelemetrySummary + Literature evidence + available refs.
    Output: Diagnosis dict that downstream task agents share.
    """

    @property
    def name(self) -> str:
        return "Diagnostician"

    def run(
        self,
        telemetry_summary: Dict[str, Any],
        literature: List[Dict[str, Any]],
        available_refs: List[str],
        seed: Optional[int] = None,
    ) -> AgentResult:
        """Run diagnosis synthesis.

        Args:
            telemetry_summary: Output from TelemetryAnalyst.run().
            literature: List of evidence dicts [{id, text, source}, ...].
            available_refs: List of valid reference IDs (for citation guidance).
            seed: LLM seed for reproducibility.

        Returns:
            AgentResult with output containing the Diagnosis.
        """
        user_prompt = DIAGNOSTICIAN_USER_PROMPT_TEMPLATE.format(
            telemetry_summary=str(telemetry_summary),
            literature=str(literature) if literature else "No literature evidence available.",
            available_refs=str(sorted(available_refs)),
        )

        try:
            parsed = self._call_llm(
                system=DIAGNOSTICIAN_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=800,
                seed=seed,
            )

            if parsed.get("parse_error"):
                parsed = self._fallback_diagnosis(telemetry_summary)

            return AgentResult(
                agent_name=self.name,
                output=parsed,
                success=True,
            )

        except Exception as e:
            fallback = self._fallback_diagnosis(telemetry_summary)
            return AgentResult(
                agent_name=self.name,
                output=fallback,
                success=True,
                error=f"LLM call failed ({e}), used fallback",
            )

    def _fallback_diagnosis(self, telemetry_summary: Dict) -> Dict:
        """Build minimal diagnosis from telemetry summary alone (no LLM)."""
        health = telemetry_summary.get("drive_health_class", "unknown")
        signals = telemetry_summary.get("critical_signals", [])

        factors = []
        for sig in signals:
            factors.append({
                "factor": sig.get("finding", sig.get("attribute", "unknown")),
                "evidence": [sig.get("ref", sig.get("id", ""))],
                "severity": "high" if sig.get("severity") == "critical" else
                           "medium" if sig.get("severity") == "degrading" else "low",
            })

        return {
            "health_state": health,
            "risk_score": {"healthy": 0.1, "watch": 0.3, "degrading": 0.6, "critical": 0.9}.get(health, 0.5),
            "primary_failure_mode": None,
            "contributing_factors": factors,
            "estimated_ttf_days": None,
            "confidence": 0.4,
            "uncertainties": ["diagnosis generated from fallback (no LLM interpretation)"],
            "diagnosis_summary": f"Drive classified as {health} based on {len(signals)} concerning signals.",
        }

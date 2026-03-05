#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-aware prompt templates for task agents that receive a Diagnosis.

In the AGENTIC pipeline, task agents don't receive raw IR — they receive
a pre-synthesized Diagnosis from the Diagnostician. These prompts are
structured to work with that richer, pre-interpreted context.

The key difference from the baseline templates:
  - Baseline: dumps entire IR blob + raw lit evidence + asks for everything
  - Agentic: provides a focused Diagnosis + curated evidence + asks for task-specific output
"""

from __future__ import annotations
from typing import Any, Dict, List


def agent_predictive_prompt(sample_id: str, diagnosis: Dict[str, Any],
                            available_refs: List[str]) -> str:
    return f"""Task: Predictive analysis for one SSD window.
You have a pre-synthesized diagnosis from the Diagnostician agent.

Sample ID: {sample_id}
Diagnosis: {diagnosis}
Available reference IDs you may cite: {available_refs}

Produce JSON with:
{{
  "task": "predictive",
  "sample_id": "{sample_id}",
  "predicted_failure": <0|1>,
  "predicted_ttf_days": <number|null>,
  "predicted_tail_latency_ms": <number|null>,
  "rationale": <short text>,
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Base predicted_failure on the diagnosis risk_score and contributing factors.
- Every atomic claim MUST cite at least one ref_id from the available refs.
- Use the diagnosis confidence to calibrate your prediction confidence.
"""


def agent_descriptive_prompt(sample_id: str, diagnosis: Dict[str, Any],
                             telemetry_summary: Dict[str, Any],
                             available_refs: List[str]) -> str:
    return f"""Task: Descriptive analysis for one SSD window.
You have a diagnosis and telemetry summary from upstream agents.

Sample ID: {sample_id}
Diagnosis: {diagnosis}
Telemetry Summary: {telemetry_summary}
Available reference IDs: {available_refs}

Produce JSON with:
{{
  "task": "descriptive",
  "sample_id": "{sample_id}",
  "summary": <text describing health + performance>,
  "key_risks": [<text>, ...],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Summary should synthesize the diagnosis into readable language.
- Key risks should come from the diagnosis contributing_factors.
- Every atomic claim MUST cite at least one valid ref_id.
"""


def agent_prescriptive_prompt(sample_id: str, diagnosis: Dict[str, Any],
                              literature: List[Dict[str, Any]],
                              available_refs: List[str]) -> str:
    return f"""Task: Prescriptive analysis (recommended actions) for one SSD window.
You have a diagnosis and literature evidence.

Sample ID: {sample_id}
Diagnosis: {diagnosis}
Literature Evidence: {literature}
Available reference IDs: {available_refs}

Produce JSON with:
{{
  "task": "prescriptive",
  "sample_id": "{sample_id}",
  "recommendations": [
    {{
      "action": <text>,
      "priority": <"low"|"med"|"high">,
      "justification": <text>,
      "support": [<ref_id>, ...]
    }}
  ],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Recommendations should be feasible operational actions.
- Use IR refs (AF_...) for device-specific justifications.
- Use LIT refs for general mechanism support.
- Prioritize based on the diagnosis risk_score and contributing factor severity.
"""


def agent_whatif_prompt(sample_id: str, diagnosis: Dict[str, Any],
                        scenario: str, literature: List[Dict[str, Any]],
                        available_refs: List[str],
                        evaluator_feedback: str = "") -> str:
    feedback_section = ""
    if evaluator_feedback:
        feedback_section = f"""
IMPORTANT — Previous attempt was rejected by the Evaluator. Fix these issues:
{evaluator_feedback}
"""

    return f"""Task: What-if (counterfactual) analysis for one SSD window.
You have a diagnosis and literature evidence.

Sample ID: {sample_id}
Diagnosis: {diagnosis}
Counterfactual scenario: {scenario}
Literature Evidence: {literature}
Available reference IDs: {available_refs}
{feedback_section}

Produce JSON with:
{{
  "task": "whatif",
  "sample_id": "{sample_id}",
  "scenario": "{scenario}",
  "analysis": <text>,
  "counterfactual_statements": [
    {{
      "statement": <text>,
      "variable": <text>,
      "delta": <number|null>,
      "effect": <text>,
      "effect_direction": <"increase"|"decrease"|"unclear">,
      "evidence": [<ref_id>, ...]
    }}
  ]
}}

Rules:
- EVERY counterfactual statement MUST cite at least one evidence ref.
- effect_direction MUST be "increase" or "decrease" when evidence supports it.
  Only use "unclear" when genuinely ambiguous.
- Use the diagnosis contributing_factors to anchor your reasoning.
"""

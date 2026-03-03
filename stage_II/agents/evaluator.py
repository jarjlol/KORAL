#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluator agent — validates task agent outputs BEFORE they're returned.

This agent runs AFTER each task agent (predictive/descriptive/prescriptive/what-if).
It checks the output quality and returns pass/fail with specific feedback.

What it does:
  1. Checks Faithfulness Precision (FiP): are all cited refs actually valid?
  2. Checks Counterfactual Validity (CFV) for what-if outputs.
  3. Checks structural completeness (required fields present?).
  4. Returns pass/fail + feedback that the orchestrator can use for retry.

Why this helps:
  - In the baseline, FiP/CFV are computed POST-HOC as evaluation metrics.
    The LLM never gets a chance to fix hallucinated citations.
  - The Evaluator catches bad citations BEFORE output, and the orchestrator
    can retry the task agent with feedback like "ref AF_r_999 doesn't exist."
  - This directly targets FiP improvement — the primary metric we aim to beat.

Note: This agent does NOT use LLM calls. It's purely deterministic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from stage_II.agents.base import Agent, AgentResult


@dataclass
class EvalFeedback:
    """Structured feedback from the Evaluator."""
    passed: bool
    fip_score: float
    cfv_score: float
    invalid_refs: List[str]       # refs cited but not in available_refs
    missing_fields: List[str]     # required fields that are absent
    suggestions: List[str]        # human-readable fix suggestions


class Evaluator(Agent):
    """Validates task agent outputs before they're returned.

    Pipeline position: LAST agent, runs after each task agent.
    Input: task output JSON + set of available reference IDs.
    Output: EvalFeedback (pass/fail + specific issues).

    This agent is DETERMINISTIC — no LLM calls needed.
    """

    def __init__(self, fip_threshold: float = 0.5, cfv_threshold: float = 0.4):
        # No LLM needed for this agent
        super().__init__(llm=None, temperature=0.0)
        self.fip_threshold = fip_threshold
        self.cfv_threshold = cfv_threshold

    @property
    def name(self) -> str:
        return "Evaluator"

    def run(
        self,
        task: str,
        output_json: Dict[str, Any],
        available_refs: Set[str],
    ) -> AgentResult:
        """Validate a task agent's output.

        Args:
            task: Task name ('predictive', 'descriptive', 'prescriptive', 'whatif').
            output_json: The parsed JSON output from the task agent.
            available_refs: Set of valid reference IDs (from IR + DataKG + Literature).

        Returns:
            AgentResult with output containing EvalFeedback.
        """
        fip = 0.0
        cfv = 0.0
        invalid_refs = []
        missing_fields = []
        suggestions = []

        # ── Check structural completeness ──────────────────────────────
        required = self._required_fields(task)
        for field in required:
            if field not in output_json or output_json[field] is None:
                missing_fields.append(field)

        if missing_fields:
            suggestions.append(f"Missing required fields: {', '.join(missing_fields)}")

        # ── Check FiP (Faithfulness Precision) ─────────────────────────
        claims = output_json.get("atomic_claims", [])
        if isinstance(claims, list) and len(claims) > 0:
            supported = 0
            total = 0
            for c in claims:
                if not isinstance(c, dict):
                    continue
                total += 1
                sup = c.get("support", [])
                if not isinstance(sup, list) or len(sup) == 0:
                    suggestions.append(
                        f"Claim '{str(c.get('claim', ''))[:50]}...' has no evidence citations."
                    )
                    continue

                # Check each cited ref
                claim_ok = True
                for ref in sup:
                    if ref is None:
                        claim_ok = False
                        continue
                    r = str(ref).strip()
                    # Strip common prefixes for matching
                    r_clean = r.replace("IR:", "").replace("ENV:", "").replace("LIT:", "").strip()

                    # Check if ref is valid
                    if (r_clean not in available_refs
                            and r not in available_refs
                            and not r_clean.startswith("LIT_")):
                        invalid_refs.append(r)
                        claim_ok = False
                if claim_ok:
                    supported += 1

            fip = float(supported / total) if total > 0 else 0.0

            if invalid_refs:
                suggestions.append(
                    f"Invalid references cited (not in DataKG/IR/Literature): "
                    f"{', '.join(set(invalid_refs))}. "
                    f"Use only refs from the provided IR (e.g., AF_r_194) or Literature (e.g., LIT_1)."
                )
        else:
            suggestions.append("No atomic_claims found in output. Add grounded claims with evidence.")

        # ── Check CFV (Counterfactual Validity) — only for what-if ────
        if task == "whatif":
            stmts = output_json.get("counterfactual_statements", [])
            if isinstance(stmts, list) and len(stmts) > 0:
                good = 0
                total_stmts = 0
                for s in stmts:
                    if not isinstance(s, dict):
                        continue
                    total_stmts += 1
                    ev = s.get("evidence", [])
                    direction = str(s.get("effect_direction", "unclear")).lower()

                    if not isinstance(ev, list) or len(ev) == 0:
                        suggestions.append(
                            f"Counterfactual statement '{str(s.get('statement', ''))[:50]}...' "
                            f"has no evidence."
                        )
                    elif direction in ("increase", "decrease"):
                        good += 1
                    else:
                        suggestions.append(
                            f"Counterfactual direction is 'unclear' — try to be specific."
                        )

                cfv = float(good / total_stmts) if total_stmts > 0 else 0.0
            else:
                suggestions.append("No counterfactual_statements found in what-if output.")

        # ── Pass/fail decision ─────────────────────────────────────────
        passed = True
        if fip < self.fip_threshold and len(claims) > 0:
            passed = False
            suggestions.append(f"FiP={fip:.2f} is below threshold {self.fip_threshold}.")
        if task == "whatif" and cfv < self.cfv_threshold:
            passed = False
            suggestions.append(f"CFV={cfv:.2f} is below threshold {self.cfv_threshold}.")
        if missing_fields:
            passed = False

        feedback = {
            "passed": passed,
            "fip_score": fip,
            "cfv_score": cfv,
            "invalid_refs": list(set(invalid_refs)),
            "missing_fields": missing_fields,
            "suggestions": suggestions,
        }

        return AgentResult(
            agent_name=self.name,
            output=feedback,
            success=True,
        )

    def _required_fields(self, task: str) -> List[str]:
        """Return required fields for each task type."""
        base = ["task", "sample_id", "atomic_claims"]
        if task == "predictive":
            return base + ["predicted_failure", "rationale"]
        elif task == "descriptive":
            return base + ["summary", "key_risks"]
        elif task == "prescriptive":
            return base + ["recommendations"]
        elif task == "whatif":
            return ["task", "sample_id", "counterfactual_statements", "analysis"]
        return base

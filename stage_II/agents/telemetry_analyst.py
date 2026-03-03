#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Telemetry Analyst agent — interprets raw SMART IR into a structured health summary.

This is the FIRST agent in the pipeline. It replaces the implicit statistical
reasoning that the single-pass LLM had to do when receiving the raw IR blob.

What it does:
  1. Receives the raw IR dict (list of SMART attribute frames, each with
     median, p95, slope, changepoints, outliers, coverage).
  2. Sends a focused prompt to the LLM asking it to interpret these statistics.
  3. Returns a structured TelemetrySummary with:
     - Per-attribute severity classification (normal/watch/degrading/critical)
     - Cross-attribute correlations
     - Overall drive health class
     - Data quality flags

Why this helps:
  - The downstream Diagnostician and Task agents receive a PRE-INTERPRETED
    summary instead of raw numbers, so they can focus on reasoning.
  - The specialized prompt is much shorter and more focused than the
    monolithic baseline prompt, leading to better statistical interpretation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from stage_II.agents.base import Agent, AgentResult


# ── Signal severity classifier (deterministic pre-filter) ──────────────

def _classify_signal(attr: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic pre-classification of a single SMART attribute frame.

    This runs BEFORE the LLM — it flags obviously concerning signals
    so the LLM can focus on interpreting ambiguous ones.

    Heuristic rules (tunable thresholds):
      - slope > 0.1 per day on error-related attrs → degrading
      - outliers > 2 → watch
      - changepoint exists and slope > 0 → degrading
      - p95/median ratio > 3 → anomalous spread
    """
    name = attr.get("attribute", "")
    slope = attr.get("slope") or 0
    outliers = attr.get("outliers", 0)
    cp = attr.get("changepoint_idx")
    median = attr.get("median") or 0
    p95 = attr.get("p95") or 0
    coverage = attr.get("coverage", 0)

    severity = "normal"
    reasons = []

    # Error-related attributes (uncorrectable errors, media errors, etc.)
    error_attrs = {"r_187", "r_188", "r_197", "r_198", "r_199", "r_5"}
    # Wear-related attributes
    wear_attrs = {"r_177", "r_173", "r_233", "r_241", "r_242"}
    # Temperature
    temp_attrs = {"r_194", "r_190", "r_231"}

    abs_slope = abs(slope)

    # Rule 1: Increasing trend on error attributes
    if name in error_attrs and slope > 0.05:
        severity = "degrading"
        reasons.append(f"increasing trend (slope={slope:.3f}/day)")
    elif name in error_attrs and slope > 0.01:
        severity = "watch"
        reasons.append(f"slight upward trend (slope={slope:.3f}/day)")

    # Rule 2: Significant outliers
    if outliers >= 3:
        if severity == "normal":
            severity = "watch"
        reasons.append(f"{outliers} outliers detected")

    # Rule 3: Changepoint + positive slope = degradation signal
    if cp is not None and slope > 0 and name in error_attrs:
        severity = "degrading"
        reasons.append(f"changepoint at day {cp} with upward trend")

    # Rule 4: High wear on wear attributes
    if name in wear_attrs and abs_slope > 0.1:
        if severity == "normal":
            severity = "watch"
        reasons.append(f"wear progressing (slope={slope:.3f}/day)")

    # Rule 5: Temperature anomalies
    if name in temp_attrs and p95 > 70:
        severity = "degrading" if p95 > 80 else "watch"
        reasons.append(f"high temperature (p95={p95:.1f}°C)")

    # Rule 6: Large spread (p95/median ratio)
    if median > 0 and p95 / median > 3 and name not in temp_attrs:
        if severity == "normal":
            severity = "watch"
        reasons.append(f"high variability (p95/median={p95/median:.1f}x)")

    # Rule 7: Low coverage = unreliable data
    if coverage < 0.5:
        reasons.append(f"low coverage ({coverage:.0%})")

    return {
        "attribute": name,
        "id": attr.get("id", f"AF_{name}"),
        "severity": severity,
        "reasons": reasons,
        "stats": {
            "median": median,
            "p95": p95,
            "slope": slope,
            "outliers": outliers,
            "changepoint_idx": cp,
            "coverage": coverage,
        },
    }


def _find_correlations(classified: List[Dict]) -> List[Dict]:
    """Find cross-attribute correlations worth mentioning.

    Simple heuristic: if two non-normal attributes coexist, flag them.
    """
    concerning = [c for c in classified if c["severity"] != "normal"]
    correlations = []

    # Temperature + errors correlation
    temp_signals = [c for c in concerning if c["attribute"] in {"r_194", "r_190", "r_231"}]
    error_signals = [c for c in concerning if c["attribute"] in {"r_187", "r_188", "r_197", "r_198", "r_199", "r_5"}]

    if temp_signals and error_signals:
        correlations.append({
            "pair": [temp_signals[0]["attribute"], error_signals[0]["attribute"]],
            "interpretation": "thermal stress may be contributing to error rate increase",
        })

    # Wear + errors correlation
    wear_signals = [c for c in concerning if c["attribute"] in {"r_177", "r_173", "r_233"}]
    if wear_signals and error_signals:
        correlations.append({
            "pair": [wear_signals[0]["attribute"], error_signals[0]["attribute"]],
            "interpretation": "wear progression may be increasing error susceptibility",
        })

    return correlations


# ── The Agent ──────────────────────────────────────────────────────────

TELEMETRY_SYSTEM_PROMPT = """You are the Telemetry Analyst agent in the KORAL SSD analysis pipeline.
You receive pre-classified SMART attribute signals and must produce a JSON health summary.
Your job is to INTERPRET the statistical signals, NOT to predict failure (that's a downstream agent's job).
Return ONLY valid JSON, no markdown fences."""

TELEMETRY_USER_PROMPT_TEMPLATE = """Analyze these pre-classified SMART telemetry signals for one SSD window.

Pre-classified signals (with deterministic severity labels):
{classified_signals}

Cross-attribute correlations detected:
{correlations}

Environmental context: {env_summary}
Workload context: {workload_summary}

Produce JSON with:
{{
  "drive_health_class": <"healthy"|"watch"|"degrading"|"critical">,
  "health_rationale": <1-2 sentence explanation>,
  "critical_signals": [
    {{
      "attribute": <string>,
      "ref": <string, e.g., "AF_r_187">,
      "finding": <text>,
      "severity": <"normal"|"watch"|"degrading"|"critical">,
      "confidence": <0.0-1.0>
    }}
  ],
  "cross_correlations": [
    {{
      "attributes": [<string>, <string>],
      "interpretation": <text>
    }}
  ],
  "data_quality_flags": [<string>, ...]
}}

Rules:
- Only include signals with severity != "normal" in critical_signals.
- If all signals are normal, set drive_health_class to "healthy".
- Each critical_signal MUST include the ref ID (e.g., "AF_r_187").
- Be specific about what the numbers mean for SSD health.
"""


class TelemetryAnalyst(Agent):
    """Interprets raw SMART IR into a structured health summary.

    Pipeline position: FIRST agent, runs after build_smart_ir().
    Input: IR dict with 'smart' key (list of attribute frames).
    Output: TelemetrySummary with classified signals and health assessment.
    """

    @property
    def name(self) -> str:
        return "TelemetryAnalyst"

    def run(
        self,
        ir: Dict[str, Any],
        env_ir: Optional[Dict[str, Any]] = None,
        workload_ir: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> AgentResult:
        """Run telemetry analysis.

        Args:
            ir: Full IR dict, must contain 'smart' key with list of attribute frames.
            env_ir: Optional environmental IR (from build_env_ir).
            workload_ir: Optional workload IR (from build_workload_ir).
            seed: LLM seed for reproducibility.

        Returns:
            AgentResult with output containing the TelemetrySummary.
        """
        smart_frames = ir.get("smart", [])

        if not smart_frames:
            return AgentResult(
                agent_name=self.name,
                output={"drive_health_class": "unknown", "critical_signals": [],
                        "cross_correlations": [], "data_quality_flags": ["no SMART data"]},
                success=True,
            )

        # Step 1: Deterministic pre-classification (no LLM needed)
        classified = [_classify_signal(af) for af in smart_frames]

        # Step 2: Find cross-attribute correlations
        correlations = _find_correlations(classified)

        # Step 3: Summarize env/workload context
        env_summary = "none"
        if env_ir and isinstance(env_ir, dict) and env_ir.get("env"):
            env_summary = str(env_ir["env"])

        workload_summary = "none"
        if workload_ir and isinstance(workload_ir, dict) and workload_ir.get("workload"):
            workload_summary = str(workload_ir["workload"])

        # Step 4: LLM interprets the pre-classified signals
        # Only send non-normal signals + a count of normal ones to save tokens
        concerning = [c for c in classified if c["severity"] != "normal"]
        normal_count = len(classified) - len(concerning)

        signals_summary = {
            "total_attributes": len(classified),
            "normal_count": normal_count,
            "concerning_signals": concerning,
        }

        user_prompt = TELEMETRY_USER_PROMPT_TEMPLATE.format(
            classified_signals=str(signals_summary),
            correlations=str(correlations) if correlations else "none detected",
            env_summary=env_summary,
            workload_summary=workload_summary,
        )

        try:
            parsed = self._call_llm(
                system=TELEMETRY_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=700,
                seed=seed,
            )

            if parsed.get("parse_error"):
                # Fallback: build summary from deterministic classification
                parsed = self._fallback_summary(classified, correlations)

            return AgentResult(
                agent_name=self.name,
                output=parsed,
                success=True,
            )

        except Exception as e:
            # If LLM fails, use deterministic fallback
            fallback = self._fallback_summary(classified, correlations)
            return AgentResult(
                agent_name=self.name,
                output=fallback,
                success=True,
                error=f"LLM call failed ({e}), used deterministic fallback",
            )

    def _fallback_summary(self, classified: List[Dict], correlations: List[Dict]) -> Dict:
        """Build a summary purely from deterministic classification (no LLM)."""
        concerning = [c for c in classified if c["severity"] != "normal"]

        if not concerning:
            health = "healthy"
        elif any(c["severity"] == "critical" for c in concerning):
            health = "critical"
        elif any(c["severity"] == "degrading" for c in concerning):
            health = "degrading"
        else:
            health = "watch"

        return {
            "drive_health_class": health,
            "health_rationale": f"{len(concerning)} concerning signals detected out of {len(classified)}",
            "critical_signals": [
                {
                    "attribute": c["attribute"],
                    "ref": c["id"],
                    "finding": "; ".join(c["reasons"]) if c["reasons"] else c["severity"],
                    "severity": c["severity"],
                    "confidence": 0.7,
                }
                for c in concerning
            ],
            "cross_correlations": correlations,
            "data_quality_flags": [
                f"{c['attribute']} low coverage"
                for c in classified if c["stats"].get("coverage", 1) < 0.5
            ],
        }

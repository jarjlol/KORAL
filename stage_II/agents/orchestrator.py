#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator — sequences agent calls and manages retries.

This is the entry point for the agentic pipeline. It replaces the direct
LLM call loop in pipeline.py (lines 172-212) when --agentic mode is used.

Flow:
  1. Telemetry Analyst → interprets SMART IR
  2. Literature Retrieval → query terms derived from analyst findings
  3. Diagnostician → synthesizes telemetry + literature into Diagnosis
  4. For each task: call task-specific LLM with Diagnosis as context
  5. Evaluator checks output → retry if below threshold
  6. Return final outputs in the SAME format as baseline

The output format is identical to baseline so that existing _aggregate()
evaluation works without ANY changes — vanilla KORAL is not touched.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set

from stage_II.agents.telemetry_analyst import TelemetryAnalyst
from stage_II.agents.diagnostician import Diagnostician
from stage_II.agents.evaluator import Evaluator
from stage_II.agents.base import AgentResult
from stage_II.llm.openai_client import OpenAIChatClient
from stage_II.prompts.agent_prompts import (
    agent_predictive_prompt,
    agent_descriptive_prompt,
    agent_prescriptive_prompt,
    agent_whatif_prompt,
)
from stage_II.utils.json_utils import extract_json_object

# Optional: for targeted literature retrieval
try:
    from stage_II.kg.literature_kg import LiteratureKG
except ImportError:
    LiteratureKG = None


TASK_SYSTEM_PROMPT = """You are a task agent in the KORAL SSD analysis pipeline.
You receive a pre-synthesized diagnosis and must produce task-specific output.
Return ONLY valid JSON, no markdown fences.
If something is unknown, use null and explain in notes or uncertainty fields."""

# SMART ID → human-readable term mapping for literature queries
SMART_ID_TO_QUERY_TERM = {
    "r_1": "critical warning",
    "r_5": "reallocated sectors",
    "r_9": "power-on hours",
    "r_12": "power cycle",
    "r_173": "wear leveling",
    "r_175": "error log",
    "r_177": "wear leveling count",
    "r_187": "uncorrectable errors",
    "r_188": "command timeout",
    "r_190": "temperature",
    "r_192": "unsafe shutdown",
    "r_194": "temperature",
    "r_195": "thermal throttle",
    "r_196": "critical temperature",
    "r_197": "pending sectors",
    "r_198": "offline uncorrectable",
    "r_199": "media errors",
    "r_231": "SSD life remaining",
    "r_232": "available spare",
    "r_233": "media wearout",
    "r_241": "total LBAs written",
    "r_242": "total LBAs read",
    "r_246": "host read commands",
    "r_247": "host write commands",
}


class Orchestrator:
    """Sequences agent calls for one sample across multiple tasks.

    Usage:
        orc = Orchestrator(llm_client, max_retries=2)
        results = orc.run_sample(ir, tasks, sample_id, lit_evidence, available_refs)
        # results is a list of dicts, same format as baseline pipeline
    """

    def __init__(
        self,
        llm: OpenAIChatClient,
        temperature: float = 0.2,
        max_retries: int = 2,
        fip_threshold: float = 0.5,
        cfv_threshold: float = 0.4,
        lit_retriever: Optional[Any] = None,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries
        self.lit_retriever = lit_retriever  # LiteratureKG instance (optional)

        # Initialize agents
        self.analyst = TelemetryAnalyst(llm=llm, temperature=temperature)
        self.diagnostician = Diagnostician(llm=llm, temperature=temperature)
        self.evaluator = Evaluator(fip_threshold=fip_threshold, cfv_threshold=cfv_threshold)

    @staticmethod
    def _extract_query_terms(telemetry_summary: Dict[str, Any]) -> List[str]:
        """Extract targeted literature query terms from the Analyst's findings.

        Instead of generic terms like ['SMART', 'SSD', 'wear'], this produces
        terms specific to what the Analyst actually found, e.g.:
          ['uncorrectable errors', 'temperature', 'thermal stress', 'NAND wear']
        """
        terms = set()

        # Always include baseline terms
        terms.add("SSD")

        # Extract terms from critical signals
        for sig in telemetry_summary.get("critical_signals", []):
            attr = sig.get("attribute", "")
            if attr in SMART_ID_TO_QUERY_TERM:
                terms.add(SMART_ID_TO_QUERY_TERM[attr])

            severity = sig.get("severity", "")
            if severity in ("degrading", "critical"):
                terms.add("failure")
                terms.add("degradation")

        # Extract terms from cross-correlations
        for corr in telemetry_summary.get("cross_correlations", []):
            interp = corr.get("interpretation", "").lower()
            if "thermal" in interp or "temperature" in interp:
                terms.add("thermal stress")
                terms.add("NAND wear")
            if "wear" in interp:
                terms.add("wear leveling")
                terms.add("endurance")

        # Extract from health class
        health = telemetry_summary.get("drive_health_class", "")
        if health in ("degrading", "critical"):
            terms.add("failure prediction")
            terms.add("remaining useful life")
        elif health == "watch":
            terms.add("early warning")

        # Extract from data quality flags
        for flag in telemetry_summary.get("data_quality_flags", []):
            if "coverage" in flag.lower():
                terms.add("data quality")

        return sorted(terms)

    def run_sample(
        self,
        ir: Dict[str, Any],
        tasks: List[str],
        sample_id: str,
        sample_payload: Dict[str, Any],
        lit_evidence: List[Dict[str, Any]],
        available_refs: Set[str],
        whatif_scenario: str = "",
        seed: int = 7,
    ) -> List[Dict[str, Any]]:
        """Run the full agentic pipeline for one sample across all tasks.

        Returns a list of result dicts (one per task) in the SAME format
        as the baseline pipeline, so _aggregate() works unchanged.
        """
        results = []
        current_seed = seed

        # ── Stage 1: Telemetry Analyst ─────────────────────────────────
        analyst_result = self.analyst.run(
            ir=ir,
            env_ir={"env": ir.get("env")} if "env" in ir else None,
            workload_ir={"workload": ir.get("workload")} if "workload" in ir else None,
            seed=current_seed,
        )
        current_seed += 1
        telemetry_summary = analyst_result.output

        # ── Stage 1.5: Targeted Literature Retrieval ──────────────────
        # Extract query terms from what the Analyst ACTUALLY found,
        # not generic terms.  Falls back to the generic lit_evidence
        # if no retriever is available.
        targeted_terms = self._extract_query_terms(telemetry_summary)

        if self.lit_retriever is not None and hasattr(self.lit_retriever, 'retrieve'):
            targeted_lit = self.lit_retriever.retrieve(targeted_terms, limit=8)
            targeted_lit_payload = [
                {"id": e.id, "text": e.text, "source": e.source}
                for e in targeted_lit
            ]
            # Merge with generic literature, deduplicating by ID
            seen_ids = {e["id"] for e in targeted_lit_payload}
            for e in lit_evidence:
                if e.get("id") not in seen_ids:
                    targeted_lit_payload.append(e)
            lit_for_diagnosis = targeted_lit_payload
        else:
            # No retriever available — use the generic literature as-is
            lit_for_diagnosis = lit_evidence

        # Update available refs with any new literature IDs
        for e in lit_for_diagnosis:
            available_refs.add(e.get("id", ""))

        # ── Stage 2: Diagnostician ─────────────────────────────────────
        diag_result = self.diagnostician.run(
            telemetry_summary=telemetry_summary,
            literature=lit_for_diagnosis,
            available_refs=sorted(list(available_refs)),
            seed=current_seed,
        )
        current_seed += 1
        diagnosis = diag_result.output

        # ── Stage 3: Task agents (with Evaluator retry loop) ──────────
        available_refs_list = sorted(list(available_refs))

        for task in tasks:
            task_output = self._run_task_with_retry(
                task=task,
                sample_id=sample_id,
                diagnosis=diagnosis,
                telemetry_summary=telemetry_summary,
                lit_evidence=lit_evidence,
                available_refs=available_refs,
                available_refs_list=available_refs_list,
                whatif_scenario=whatif_scenario,
                seed=current_seed,
            )
            current_seed += 1

            results.append({
                "sample_id": sample_id,
                "task": task,
                "response_json": task_output,
                # Include agent trace for debugging/analysis
                "agent_trace": {
                    "telemetry_analyst": {
                        "health_class": telemetry_summary.get("drive_health_class"),
                        "num_critical_signals": len(telemetry_summary.get("critical_signals", [])),
                    },
                    "diagnostician": {
                        "health_state": diagnosis.get("health_state"),
                        "risk_score": diagnosis.get("risk_score"),
                        "confidence": diagnosis.get("confidence"),
                    },
                },
            })

            # polite pacing between API calls
            time.sleep(0.2)

        return results

    def _run_task_with_retry(
        self,
        task: str,
        sample_id: str,
        diagnosis: Dict[str, Any],
        telemetry_summary: Dict[str, Any],
        lit_evidence: List[Dict[str, Any]],
        available_refs: Set[str],
        available_refs_list: List[str],
        whatif_scenario: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single task agent with Evaluator-driven retry loop."""

        evaluator_feedback = ""

        for attempt in range(1 + self.max_retries):
            # Build task-specific prompt
            prompt = self._build_task_prompt(
                task=task,
                sample_id=sample_id,
                diagnosis=diagnosis,
                telemetry_summary=telemetry_summary,
                lit_evidence=lit_evidence,
                available_refs_list=available_refs_list,
                whatif_scenario=whatif_scenario,
                evaluator_feedback=evaluator_feedback,
            )

            # Call LLM
            resp = self.llm.chat(
                system=TASK_SYSTEM_PROMPT,
                user=prompt,
                temperature=self.temperature,
                max_tokens=900,
                seed=seed + attempt,
            )

            parsed = extract_json_object(resp.text) or {
                "task": task,
                "sample_id": sample_id,
                "parse_error": True,
                "raw_text": resp.text,
            }

            # Run Evaluator
            eval_result = self.evaluator.run(
                task=task,
                output_json=parsed,
                available_refs=available_refs,
            )
            eval_feedback = eval_result.output

            if eval_feedback["passed"] or attempt == self.max_retries:
                # Store evaluation scores in the output for metrics
                if task in ("descriptive", "prescriptive"):
                    parsed["FiP"] = eval_feedback["fip_score"]
                if task == "whatif":
                    parsed["CFV"] = eval_feedback["cfv_score"]
                return parsed

            # Retry: format evaluator feedback for next attempt
            evaluator_feedback = "\n".join(eval_feedback["suggestions"])

        return parsed  # shouldn't reach here, but just in case

    def _build_task_prompt(
        self,
        task: str,
        sample_id: str,
        diagnosis: Dict,
        telemetry_summary: Dict,
        lit_evidence: List[Dict],
        available_refs_list: List[str],
        whatif_scenario: str,
        evaluator_feedback: str,
    ) -> str:
        """Build the appropriate prompt for each task type."""

        if task == "predictive":
            return agent_predictive_prompt(
                sample_id, diagnosis, available_refs_list,
            )
        elif task == "descriptive":
            return agent_descriptive_prompt(
                sample_id, diagnosis, telemetry_summary, available_refs_list,
            )
        elif task == "prescriptive":
            return agent_prescriptive_prompt(
                sample_id, diagnosis, lit_evidence, available_refs_list,
            )
        elif task == "whatif":
            return agent_whatif_prompt(
                sample_id, diagnosis, whatif_scenario,
                lit_evidence, available_refs_list, evaluator_feedback,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

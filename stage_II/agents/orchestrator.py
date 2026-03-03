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


TASK_SYSTEM_PROMPT = """You are a task agent in the KORAL SSD analysis pipeline.
You receive a pre-synthesized diagnosis and must produce task-specific output.
Return ONLY valid JSON, no markdown fences.
If something is unknown, use null and explain in notes or uncertainty fields."""


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
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize agents
        self.analyst = TelemetryAnalyst(llm=llm, temperature=temperature)
        self.diagnostician = Diagnostician(llm=llm, temperature=temperature)
        self.evaluator = Evaluator(fip_threshold=fip_threshold, cfv_threshold=cfv_threshold)

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

        # ── Stage 2: Diagnostician ─────────────────────────────────────
        diag_result = self.diagnostician.run(
            telemetry_summary=telemetry_summary,
            literature=lit_evidence,
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

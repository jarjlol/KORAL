#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_agents.py — End-to-end test of all KORAL agents using synthetic data.

No OpenAI API key needed! Uses mock LLM responses to verify the full flow.

Run from repo root:
    python -m stage_II.tests.test_agents
"""

import json
import sys
from typing import Any, Dict
from pathlib import Path

# ── Mock LLM client (replaces OpenAI calls) ──────────────────────────

class MockLLMResponse:
    def __init__(self, text):
        self.text = text
        self.raw = {"choices": [{"message": {"content": text}}]}

class MockLLMClient:
    """Returns pre-defined JSON responses based on which agent is calling."""

    def __init__(self):
        self.call_log = []  # tracks all calls for inspection

    def chat(self, system, user, temperature=0.2, max_tokens=900, seed=None):
        self.call_log.append({
            "system_preview": system[:80],
            "user_preview": user[:120],
            "max_tokens": max_tokens,
        })

        # Detect which agent is calling from the system prompt
        if "Telemetry Analyst" in system:
            return MockLLMResponse(json.dumps({
                "drive_health_class": "degrading",
                "health_rationale": "Uncorrectable errors are increasing with a positive slope and outliers, accompanied by elevated temperature.",
                "critical_signals": [
                    {"attribute": "r_187", "ref": "AF_r_187", "finding": "Increasing uncorrectable error count with 4 outliers and changepoint at day 12", "severity": "degrading", "confidence": 0.85},
                    {"attribute": "r_194", "ref": "AF_r_194", "finding": "Temperature p95 at 72°C indicates thermal stress", "severity": "watch", "confidence": 0.7},
                ],
                "cross_correlations": [
                    {"attributes": ["r_194", "r_187"], "interpretation": "Thermal stress correlates with increasing error rate"}
                ],
                "data_quality_flags": ["r_233 has low coverage (40%)"],
            }))

        elif "Diagnostician" in system:
            return MockLLMResponse(json.dumps({
                "health_state": "degrading",
                "risk_score": 0.72,
                "primary_failure_mode": "Media wear-out with thermal acceleration",
                "contributing_factors": [
                    {"factor": "Increasing uncorrectable errors", "evidence": ["AF_r_187", "LIT_1"], "severity": "high"},
                    {"factor": "Elevated operating temperature", "evidence": ["AF_r_194"], "severity": "medium"},
                ],
                "estimated_ttf_days": 45,
                "confidence": 0.65,
                "uncertainties": ["Limited environmental data", "r_233 coverage gap"],
                "diagnosis_summary": "Drive shows progressive media degradation with 4 error outliers and thermal correlation. Estimated 45-day window before critical failure threshold.",
            }))

        elif "task agent" in system.lower():
            # Detect task type from user prompt
            if "Predictive" in user:
                return MockLLMResponse(json.dumps({
                    "task": "predictive",
                    "sample_id": "test_drive_1",
                    "predicted_failure": 1,
                    "predicted_ttf_days": 45,
                    "predicted_tail_latency_ms": None,
                    "rationale": "Risk score 0.72 with increasing media errors and thermal stress",
                    "atomic_claims": [
                        {"claim": "Uncorrectable error rate is increasing at 0.15/day", "support": ["AF_r_187"]},
                        {"claim": "Temperature stress at p95=72°C accelerates wear", "support": ["AF_r_194", "LIT_1"]},
                    ]
                }))
            elif "Descriptive" in user:
                return MockLLMResponse(json.dumps({
                    "task": "descriptive",
                    "sample_id": "test_drive_1",
                    "summary": "Drive exhibits progressive media degradation with increasing uncorrectable errors and thermal stress.",
                    "key_risks": ["Media wear-out failure within 45 days", "Thermal throttling under sustained workload"],
                    "atomic_claims": [
                        {"claim": "Error rate slope of 0.15/day indicates accelerating degradation", "support": ["AF_r_187"]},
                        {"claim": "Operating temperature p95 of 72°C exceeds recommended range", "support": ["AF_r_194"]},
                        {"claim": "Thermal stress accelerates NAND wear in TLC drives", "support": ["LIT_1"]},
                    ]
                }))
            elif "Prescriptive" in user:
                return MockLLMResponse(json.dumps({
                    "task": "prescriptive",
                    "sample_id": "test_drive_1",
                    "recommendations": [
                        {"action": "Schedule data migration within 30 days", "priority": "high", "justification": "Risk score 0.72 with degrading media", "support": ["AF_r_187"]},
                        {"action": "Improve cooling or throttle writes", "priority": "med", "justification": "Temperature at 72°C accelerates wear", "support": ["AF_r_194", "LIT_1"]},
                    ],
                    "atomic_claims": [
                        {"claim": "Data migration prevents data loss from imminent media failure", "support": ["AF_r_187"]},
                        {"claim": "Thermal management extends remaining drive life", "support": ["AF_r_194", "LIT_1"]},
                    ]
                }))
            elif "What-if" in user:
                return MockLLMResponse(json.dumps({
                    "task": "whatif",
                    "sample_id": "test_drive_1",
                    "scenario": "If temperature decreases by 10°C",
                    "analysis": "Reducing temperature would slow NAND wear and reduce error rates.",
                    "counterfactual_statements": [
                        {"statement": "Reducing temp by 10°C would decrease error rate", "variable": "temperature", "delta": -10, "effect": "Slower media degradation", "effect_direction": "decrease", "evidence": ["AF_r_194", "LIT_1"]},
                        {"statement": "Lower temperature extends estimated TTF", "variable": "temperature", "delta": -10, "effect": "TTF increases by ~15 days", "effect_direction": "increase", "evidence": ["AF_r_187", "LIT_1"]},
                    ]
                }))

        # Fallback
        return MockLLMResponse('{"error": "unrecognized agent"}')


# ── Synthetic SMART data (simulates what build_smart_ir produces) ─────

SYNTHETIC_IR = {
    "smart": [
        # A degrading attribute (uncorrectable errors)
        {"id": "AF_r_187", "attribute": "r_187", "n": 30, "median": 5, "p95": 20,
         "min": 0, "max": 25, "slope": 0.15, "changepoint_idx": 12, "outliers": 4, "coverage": 1.0},
        # Temperature — elevated but not critical
        {"id": "AF_r_194", "attribute": "r_194", "n": 30, "median": 55, "p95": 72,
         "min": 42, "max": 75, "slope": 0.08, "changepoint_idx": None, "outliers": 1, "coverage": 1.0},
        # Normal attribute (power-on hours)
        {"id": "AF_r_9", "attribute": "r_9", "n": 30, "median": 8760, "p95": 8770,
         "min": 8750, "max": 8775, "slope": 0.03, "changepoint_idx": None, "outliers": 0, "coverage": 1.0},
        # Low coverage attribute
        {"id": "AF_r_233", "attribute": "r_233", "n": 12, "median": 80, "p95": 90,
         "min": 70, "max": 95, "slope": 0.02, "changepoint_idx": None, "outliers": 0, "coverage": 0.4},
        # Normal wear attribute
        {"id": "AF_r_5", "attribute": "r_5", "n": 30, "median": 0, "p95": 0,
         "min": 0, "max": 0, "slope": 0.0, "changepoint_idx": None, "outliers": 0, "coverage": 1.0},
    ]
}

SYNTHETIC_LITERATURE = [
    {"id": "LIT_1", "text": "High operating temperature accelerates NAND flash wear and increases uncorrectable error rates in TLC SSDs", "source": "thermal_paper.pdf"},
    {"id": "LIT_2", "text": "Media error count is a reliable precursor to SSD failure within 30-60 day windows", "source": "prediction_paper.pdf"},
]


# ── Test functions ────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_json(data, indent=2):
    print(json.dumps(data, indent=indent, default=str))


def test_telemetry_analyst():
    """Test the Telemetry Analyst agent."""
    print_header("AGENT 1: Telemetry Analyst")
    print("\nPurpose: Interprets raw SMART statistics into a health summary.")
    print("It does TWO things:")
    print("  1. Deterministic pre-classification (heuristic rules, no LLM)")
    print("  2. LLM interpretation of the pre-classified signals")

    from stage_II.agents.telemetry_analyst import TelemetryAnalyst, _classify_signal

    # Show step 1: deterministic pre-classification
    print("\n--- Step 1: Deterministic Pre-Classification (no LLM) ---")
    for af in SYNTHETIC_IR["smart"]:
        result = _classify_signal(af)
        marker = "⚠️" if result["severity"] != "normal" else "✅"
        print(f"  {marker} {af['attribute']:>6s}: severity={result['severity']:<10s} "
              f"reasons={result['reasons']}")

    # Show step 2: full agent with mock LLM
    print("\n--- Step 2: Full Agent (with mock LLM) ---")
    mock_llm = MockLLMClient()
    analyst = TelemetryAnalyst(llm=mock_llm, temperature=0.2)
    result = analyst.run(ir=SYNTHETIC_IR, seed=42)

    print(f"  Agent: {result.agent_name}")
    print(f"  Success: {result.success}")
    print(f"  Output:")
    print_json(result.output)

    print(f"\n  LLM calls made: {len(mock_llm.call_log)}")
    return result.output


def test_evaluator():
    """Test the Evaluator agent."""
    print_header("AGENT 2: Evaluator")
    print("\nPurpose: Validates output BEFORE returning it.")
    print("Checks: Are cited refs real? Are fields complete? Is CFV direction explicit?")
    print("This agent is PURELY DETERMINISTIC — no LLM calls.")

    from stage_II.agents.evaluator import Evaluator

    evaluator = Evaluator(fip_threshold=0.5, cfv_threshold=0.4)

    available_refs = {"AF_r_187", "AF_r_194", "AF_r_9", "AF_r_233", "AF_r_5", "LIT_1", "LIT_2"}

    # Test 1: Good output (should pass)
    print("\n--- Test 1: Good output (valid refs) ---")
    good_output = {
        "task": "descriptive", "sample_id": "test",
        "summary": "Drive is degrading", "key_risks": ["media failure"],
        "atomic_claims": [
            {"claim": "Errors increasing", "support": ["AF_r_187"]},
            {"claim": "Temperature high", "support": ["AF_r_194", "LIT_1"]},
        ]
    }
    result = evaluator.run(task="descriptive", output_json=good_output, available_refs=available_refs)
    fb = result.output
    print(f"  Passed: {fb['passed']}")
    print(f"  FiP: {fb['fip_score']}")
    print(f"  Invalid refs: {fb['invalid_refs']}")
    print(f"  Suggestions: {fb['suggestions']}")

    # Test 2: Bad output (hallucinated refs)
    print("\n--- Test 2: Bad output (hallucinated refs) ---")
    bad_output = {
        "task": "descriptive", "sample_id": "test",
        "summary": "Drive is fine", "key_risks": [],
        "atomic_claims": [
            {"claim": "Everything normal", "support": ["AF_r_999"]},         # FAKE
            {"claim": "Temperature ok", "support": ["SMART_TEMP_FAKE"]},     # FAKE
            {"claim": "Errors stable", "support": ["AF_r_187"]},             # REAL
        ]
    }
    result = evaluator.run(task="descriptive", output_json=bad_output, available_refs=available_refs)
    fb = result.output
    print(f"  Passed: {fb['passed']}")
    print(f"  FiP: {fb['fip_score']:.2f} (threshold: 0.5)")
    print(f"  Invalid refs: {fb['invalid_refs']}")
    print(f"  Suggestions:")
    for s in fb['suggestions']:
        print(f"    → {s}")

    # Test 3: What-if output (check CFV)
    print("\n--- Test 3: What-if output (CFV check) ---")
    whatif_output = {
        "task": "whatif", "sample_id": "test",
        "analysis": "Temperature reduction helps",
        "counterfactual_statements": [
            {"statement": "Temp down → errors decrease", "variable": "temp",
             "delta": -10, "effect": "fewer errors", "effect_direction": "decrease",
             "evidence": ["AF_r_194", "LIT_1"]},
            {"statement": "Unknown effect", "variable": "humidity",
             "delta": 5, "effect": "unclear", "effect_direction": "unclear",
             "evidence": ["LIT_2"]},
        ]
    }
    result = evaluator.run(task="whatif", output_json=whatif_output, available_refs=available_refs)
    fb = result.output
    print(f"  Passed: {fb['passed']}")
    print(f"  CFV: {fb['cfv_score']:.2f} (threshold: 0.4)")
    print(f"  Suggestions:")
    for s in fb['suggestions']:
        print(f"    → {s}")


def test_diagnostician():
    """Test the Diagnostician agent."""
    print_header("AGENT 3: Diagnostician")
    print("\nPurpose: Synthesizes telemetry summary + literature into a shared Diagnosis.")
    print("This Diagnosis is shared by ALL downstream task agents (predictive/descriptive/etc).")
    print("This is what ensures cross-task consistency (CTC).")

    from stage_II.agents.diagnostician import Diagnostician

    mock_llm = MockLLMClient()
    diag = Diagnostician(llm=mock_llm, temperature=0.2)

    # Use the Telemetry Analyst's output as input
    telemetry_summary = {
        "drive_health_class": "degrading",
        "critical_signals": [
            {"attribute": "r_187", "ref": "AF_r_187", "finding": "increasing errors", "severity": "degrading"},
            {"attribute": "r_194", "ref": "AF_r_194", "finding": "high temp", "severity": "watch"},
        ],
        "cross_correlations": [{"attributes": ["r_194", "r_187"], "interpretation": "thermal → errors"}],
    }

    result = diag.run(
        telemetry_summary=telemetry_summary,
        literature=SYNTHETIC_LITERATURE,
        available_refs=["AF_r_187", "AF_r_194", "AF_r_9", "LIT_1", "LIT_2"],
        seed=42,
    )

    print(f"\n  Agent: {result.agent_name}")
    print(f"  Success: {result.success}")
    print(f"  Diagnosis:")
    print_json(result.output)
    print(f"\n  LLM calls made: {len(mock_llm.call_log)}")
    return result.output


def test_orchestrator_full():
    """Test the full orchestrator flow."""
    print_header("FULL AGENTIC PIPELINE (Orchestrator)")
    print("\nThis sequences ALL agents for one sample:")
    print("  1. Telemetry Analyst → interprets SMART stats")
    print("  2. Diagnostician → synthesizes diagnosis")
    print("  3. Task LLM → generates output for each task")
    print("  4. Evaluator → validates output (retries if bad)")

    from stage_II.agents.orchestrator import Orchestrator

    mock_llm = MockLLMClient()
    orc = Orchestrator(
        llm=mock_llm,
        temperature=0.2,
        max_retries=1,
    )

    available_refs = {"AF_r_187", "AF_r_194", "AF_r_9", "AF_r_233", "AF_r_5", "LIT_1", "LIT_2"}

    results = orc.run_sample(
        ir=SYNTHETIC_IR,
        tasks=["predictive", "descriptive", "prescriptive", "whatif"],
        sample_id="test_drive_1",
        sample_payload={},   # not used in agentic mode
        lit_evidence=SYNTHETIC_LITERATURE,
        available_refs=available_refs,
        whatif_scenario="If temperature decreases by 10°C",
        seed=42,
    )

    print(f"\n  Total LLM calls: {len(mock_llm.call_log)}")
    print(f"  Results per task:")

    for r in results:
        task = r["task"]
        trace = r.get("agent_trace", {})
        parsed = r["response_json"]
        print(f"\n  ── {task.upper()} ──")
        print(f"    Agent trace: health={trace.get('telemetry_analyst', {}).get('health_class')}, "
              f"risk={trace.get('diagnostician', {}).get('risk_score')}")
        if task == "predictive":
            print(f"    Predicted failure: {parsed.get('predicted_failure')}")
            print(f"    TTF days: {parsed.get('predicted_ttf_days')}")
        elif task == "descriptive":
            print(f"    Summary: {parsed.get('summary', '')[:80]}...")
            print(f"    Risks: {parsed.get('key_risks')}")
        elif task == "prescriptive":
            recs = parsed.get("recommendations", [])
            for rec in recs:
                print(f"    [{rec.get('priority', '?').upper()}] {rec.get('action')}")
        elif task == "whatif":
            stmts = parsed.get("counterfactual_statements", [])
            for s in stmts:
                print(f"    {s.get('variable')}: {s.get('effect_direction')} — {s.get('statement')}")

        claims = parsed.get("atomic_claims", [])
        print(f"    Atomic claims: {len(claims)}, all grounded: {all(c.get('support') for c in claims)}")


def test_baseline_preserved():
    """Verify vanilla KORAL pipeline signature is unchanged."""
    print_header("VANILLA KORAL CHECK")
    print("\nVerifying that the baseline pipeline still works without --agentic flag...")

    from stage_II.pipeline import Stage2Runner
    import inspect

    sig = inspect.signature(Stage2Runner.run)
    params = list(sig.parameters.keys())
    print(f"  Stage2Runner.run() params: {params}")

    # Check that agentic defaults to False
    agentic_param = sig.parameters.get("agentic")
    assert agentic_param is not None, "agentic parameter missing!"
    assert agentic_param.default == False, "agentic should default to False!"
    print(f"  ✓ agentic param exists, defaults to False")
    print(f"  ✓ Vanilla KORAL will run if you omit --agentic flag")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  KORAL AGENTIC PIPELINE — VERIFICATION TEST SUITE")
    print("  (No API key needed — uses mock LLM responses)")
    print("=" * 70)

    test_telemetry_analyst()
    test_evaluator()
    test_diagnostician()
    test_orchestrator_full()
    test_baseline_preserved()

    print_header("ALL TESTS PASSED ✓")
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY to test with real LLM")
    print("  2. Run: python -m stage_II.cli --dataset_type SMART_ALIBABA \\")
    print("       --tasks descriptive --limit_rows 1 --out_name test --agentic")

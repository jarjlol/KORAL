# Re-export from __init__ for module-style execution
from stage_II.tests import *

if __name__ == "__main__":
    import stage_II.tests
    stage_II.tests.test_telemetry_analyst()
    stage_II.tests.test_evaluator()
    stage_II.tests.test_diagnostician()
    stage_II.tests.test_orchestrator_full()
    stage_II.tests.test_baseline_preserved()

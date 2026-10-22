from pathlib import Path

REPO_PATH = Path(__file__).parent.resolve()
SAMPLE_PATH = REPO_PATH / "distributions" / "data"
FEAS_THRESH = 5e-3   # global feasibility threshold

from pathlib import Path

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns commonly categorical in NSL-KDD
NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
DEFAULT_TARGET_COL = "label"

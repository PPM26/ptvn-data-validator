import os, json
from dotenv import load_dotenv

load_dotenv()

def parse_dataset_ids(raw: str) -> list[str]:
    """
    Convert RAGFLOW_PO_DATASET_IDS to list.
    Split with comma "," will make str to list format before split and it's fine even there is only one value.
    """

    if not raw:
        return []
    raw = raw.strip()

    # JSON list format
    if raw.startswith("["):
        try:
            ids = json.loads(raw)
            return [str(x).strip() for x in ids if str(x).strip()]
        except Exception:
            pass

    return [x.strip().strip('"').strip("'") for x in raw.split(",") if x.strip()]


MODEL_URL = os.getenv("MODEL_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE"))
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

RAGFLOW_URL = os.getenv("RAGFLOW_URL")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_PO_DATASET_IDS = parse_dataset_ids(os.getenv("RAGFLOW_PO_DATASET_IDS", ""))

TOP_K = int(os.getenv("TOP_K"))
CONCURRENCY = int(os.getenv("CONCURRENCY"))


from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from app.services.llm_service import LLMService
from app.services.ragflow_service import RagFlowService
from app.services.fixer_service import FixerService


@dataclass
class AppState:
    prompts: Dict[str, str]
    llm: LLMService
    rag: RagFlowService
    fixer: FixerService


def load_prompts() -> Dict[str, str]:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"  # app/prompts
    def _read(name: str) -> str:
        return (prompts_dir / name).read_text(encoding="utf-8")

    return {
        "predict_item": _read("predict_item.txt"),
        "fix_category": _read("fix_category.txt"),
        "fix_spec": _read("fix_spec.txt"),
        "remove_multi_items": _read("remove_multi_items.txt"),
        "validate_spec": _read("validate_spec.txt"),
    }

from fastapi import APIRouter, Depends
from app.fastapi.api.deps import get_state
from app.fastapi.api.models import FixCategoryIn, FixCategoryOut
from app.fastapi.api.state import AppState

router = APIRouter()

@router.post("/fix", response_model=FixCategoryOut)
async def fix_category(payload: FixCategoryIn, state: AppState = Depends(get_state)):
    prompt = state.prompts["fix_category"]
    res = await state.llm.afix_category_structured(
        prompt,
        description=payload.description,
        category=payload.category,
        item=payload.item,
        rag_categories=payload.rag_categories,
    )
    return FixCategoryOut(category_fixed=res.category_fixed)

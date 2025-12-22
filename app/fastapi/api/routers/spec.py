from fastapi import APIRouter, Depends
from app.fastapi.api.deps import get_state
from app.fastapi.api.models import FixSpecIn, FixSpecOut, RemoveMultiItemsIn, RemoveMultiItemsOut, ValidateSpecIn, ValidateSpecOut
from app.fastapi.api.state import AppState

router = APIRouter()

@router.post("/fix", response_model=FixSpecOut)
async def fix_spec(payload: FixSpecIn, state: AppState = Depends(get_state)):
    prompt = state.prompts["fix_spec"]
    res = await state.llm.afix_spec_structured(
        prompt,
        description=payload.description,
        spec_pred=payload.spec_pred,
        item_pred=payload.item_pred,
        category_fixed=payload.category_fixed,
        spec_patterns=payload.spec_patterns,
    )
    return FixSpecOut(spec_pred_fixed=res.spec_pred_fixed)

@router.post("/remove-multi-items", response_model=RemoveMultiItemsOut)
async def remove_multi_items(payload: RemoveMultiItemsIn, state: AppState = Depends(get_state)):
    prompt = state.prompts["remove_multi_items"]
    res = await state.llm.aremove_multi_items_structured(
        prompt,
        description=payload.description,
        spec_pred_fixed=payload.spec_pred_fixed,
        category_fixed=payload.category_fixed,
    )
    return RemoveMultiItemsOut(spec_pred_remove_items=res.spec_pred_remove_items)

@router.post("/validate", response_model=ValidateSpecOut)
async def validate_spec(payload: ValidateSpecIn, state: AppState = Depends(get_state)):
    prompt = state.prompts["validate_spec"]
    res = await state.llm.avalidate_spec_structured(
        prompt,
        description=payload.description,
        spec_pred_remove_items=payload.spec_pred_remove_items,
    )
    return ValidateSpecOut(spec_pred_fixed_validated=res.spec_pred_fixed_validated)

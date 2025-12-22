from fastapi import APIRouter, Depends
from app.fastapi.api.deps import get_state
from app.fastapi.api.models import PredictItemIn, PredictItemOut
from app.fastapi.api.state import AppState

router = APIRouter()

@router.post("/predict", response_model=PredictItemOut)
async def predict_item(payload: PredictItemIn, state: AppState = Depends(get_state)):
    prompt = state.prompts["predict_item"]
    res = await state.llm.apredict_item_structured(prompt, description=payload.description)
    return PredictItemOut(item_pred=res.item_pred)

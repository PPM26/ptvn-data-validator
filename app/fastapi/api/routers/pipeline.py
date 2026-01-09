import asyncio
from typing import List, Dict, Any, Union

from fastapi import APIRouter, Depends, File, Form
from app.fastapi.api.deps import get_state
from app.fastapi.api.models import RowIn, FixRowOut, BatchFixIn, PostProcessRowOut, SingleRowFixIn
from app.fastapi.api.state import AppState
from app.utils.config import CONCURRENCY

router = APIRouter()


def fallback_result(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_pred": None,
        "item_extracted": None,
        "spec_pred_fixed": row.get("spec_pred"),
        "category_fixed": row.get("category"),
        "spec_changed": False,
        "category_changed": False,
    }


@router.post("/fix-row", response_model=Union[FixRowOut, PostProcessRowOut])
async def fix_one_row(payload: SingleRowFixIn, state: AppState = Depends(get_state)):
    row = payload.model_dump(exclude={"post_process"})
    try:
        res = await state.fixer.fix_row(row, state.prompts)
    except Exception:
        res = fallback_result(row)
    
    if payload.post_process:
        processed_res = {
            "description": payload.description,
            "category": res.get("category_fixed"),
            "spec_pred": res.get("spec_pred_fixed")
        }
        return PostProcessRowOut(**processed_res)

    return FixRowOut(**res)


@router.post("/fix-batch", response_model=Union[List[FixRowOut], List[PostProcessRowOut]])
async def fix_batch(payload: BatchFixIn, state: AppState = Depends(get_state)):
    concurrency = CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = [None] * len(payload.rows)

    async def worker(i: int, r: RowIn):
        async with semaphore:
            row = r.model_dump()
            try:
                res = await state.fixer.fix_row(row, state.prompts)
            except Exception:
                res = fallback_result(row)
            results[i] = res

    tasks = [asyncio.create_task(worker(i, r)) for i, r in enumerate(payload.rows)]
    await asyncio.gather(*tasks)

    if payload.post_process:
        processed_results = []
        for i, res in enumerate(results):
            processed_results.append({
                "description": payload.rows[i].description,
                "category": res.get("category_fixed"),
                "spec_pred": res.get("spec_pred_fixed")
            })
        return [PostProcessRowOut(**r) for r in processed_results]

    return [FixRowOut(**r) for r in results]

import asyncio
from pathlib import Path
import pandas as pd
from tqdm import tqdm 


from app.services.fixer_service import FixerService
from app.utils.config import CONCURRENCY


def load_prompt(name: str) -> str:
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (prompts_dir / name).read_text(encoding="utf-8")

def fallback_result(row):
    return {
        "item_pred": None,
        "item_extracted": None,
        "spec_pred_fixed": row.get("spec_pred"),
        "category_fixed": row.get("category"),
        "spec_changed": False,
        "category_changed": False,
    }


async def process_dataframe(df: pd.DataFrame, prompts: dict, concurrency: int) -> pd.DataFrame:
    """
    Process the DataFrame with a concurrency limit.
    """
    fixer = FixerService()
    total = len(df)

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * total

    async def worker(idx: int, row: pd.Series):
        async with semaphore:
            try:
                res = await fixer.fix_row(row, prompts)

            except asyncio.TimeoutError:
                print(f"[TIMEOUT] Row {idx}")
                res = fallback_result(row)
            except Exception as e:
                print(f"[ERROR] Row {idx} failed: {e}")
                res = fallback_result(row)
        
            return idx, res


    # Create tasks
    tasks = [
        asyncio.create_task(worker(idx, row))
        for idx, (_, row) in enumerate(df.iterrows())
    ]

    # Iterate as tasks complete, update tqdm
    for fut in tqdm(asyncio.as_completed(tasks), total=total, desc="Processing"):
        idx, res = await fut
        results[idx] = res

    # Write results back to df
    df["item_pred"] = [r["item_pred"] for r in results]
    df["item_extracted"] = [r["item_extracted"] for r in results]
    df["spec_pred_fixed"] = [r["spec_pred_fixed"] for r in results]
    df["category_fixed"] = [r["category_fixed"] for r in results]
    df["spec_changed"] = [r["spec_changed"] for r in results]
    df["category_changed"] = [r["category_changed"] for r in results]

    return df


async def main(
    input_file: str = "data/extract_false.csv",
    output_file: str = "output/dataset_fixed.csv",
    concurrency: int = CONCURRENCY,
    post_process: bool = True,
):
    root = Path(__file__).resolve().parent.parent
    input_path = root / input_file
    output_path = root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)
    df = df[19:30]
    print(f"[INFO] Loaded rows: {len(df)}")


    prompts = {
        "predict_item": load_prompt("predict_item.txt"),
        "fix_category": load_prompt("fix_category.txt"),
        "fix_spec": load_prompt("fix_spec.txt"),
        "remove_multi_items": load_prompt("remove_multi_items.txt"),
        "validate_spec": load_prompt("validate_spec.txt"),
    }

    print(f"[INFO] Running async fixes with concurrency={concurrency} ...")
    df_fixed = await process_dataframe(df, prompts, concurrency=concurrency)

    if post_process:
        df_fixed["category"] = df_fixed["category_fixed"]
        df_fixed["spec_pred"] = df_fixed["spec_pred_fixed"]
        df_fixed = df_fixed[["description", "category", "spec_pred"]]

    print(f"[INFO] Saving to: {output_path}")
    df_fixed.to_csv(output_path, index=False)
    print("[INFO] DONE!")

import asyncio
from typing import Any, Dict

import pandas as pd

from app.services.llm_service import LLMService
from app.services.ragflow_service import RagFlowService
from app.utils.spec_parser import extract_item, align_spec_keys


class FixerService:
    """
    Workflow (per row):

    1) Predict item from description -> item_pred
    2) Build spec_query = "item_pred + category" -> RagFlow spec_patterns
    3) Fix spec_pred using fix_spec prompt + spec_patterns -> spec_pred_fixed
    4) Remove multiple 'item' keys -> spec_pred_remove_items
    5) Validate spec_pred_remove_items with description -> spec_pred_fixed_validated (final_spec)
    6) Extract item_extracted from spec_pred_fixed_validated

    Returns, per row:
        - item_pred
        - item_extracted
        - spec_pred_fixed
        - category_fixed
        - spec_changed (bool)
        - category_changed (bool)
    """

    def __init__(self):
        self.llm = LLMService()
        self.rag = RagFlowService()

    # ---------- helpers ----------

    async def _safe_llm_call(
        self,
        template: str,
        timeout: float = 45.0,
        **kwargs: Any,
    ) -> str | None:
        """
        Run LLM call with timeout safeguard.
        Returns raw text (model output) or None on error/timeout.
        """
        try:
            coro = self.llm.ainvoke_prompt(template, **kwargs)
            msg = await asyncio.wait_for(coro, timeout=timeout)
            return (msg.content or "").strip()
        except asyncio.TimeoutError:
            print("[LLM] Timeout. No fix for this part.")
            return None
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None


    @staticmethod
    def _normalize_for_compare(val: Any) -> str:
        """
        Normalize values for change-detection comparison.
        NaN → empty string.
        """
        if val is None:
            return ""
        if isinstance(val, float) and pd.isna(val):
            return ""
        return str(val).strip()


    # ---------- main: fix one row ----------

    async def fix_row(self, row: Dict[str, Any], prompts: Dict[str, str]) -> Dict[str, Any]:
        """
        Workflow:

        1) Predict item from description -> item_pred
        2) Build spec_query = "item_pred + category" -> RagFlow spec_patterns
        3) Fix spec_pred using fix_spec prompt + spec_patterns -> spec_pred_fixed, spec_changed
        4) Remove multiple 'item' keys -> spec_pred_remove_items
        5) Validate spec_pred_remove_items with description -> spec_pred_fixed_validated (final_spec)
        6) Extract item_extracted from spec_pred_fixed_validated
        """
        description = row["description"]
        raw_spec_pred = row["spec_pred"]
        original_category = row["category"]

        # Normalize spec_pred for LLM input (NaN -> "")
        original_spec_pred = "" if pd.isna(raw_spec_pred) else str(raw_spec_pred)

        # ---------------- 1) Predict item from description → item_pred ----------------
        try:
            item_pred_text = await asyncio.wait_for(
                self.llm.apredict_item(
                    prompts["predict_item"],
                    description=description,
                ),
                timeout=60,
            )
            item_pred = item_pred_text.item_pred
        except asyncio.TimeoutError:
            print("[TIMEOUT] predict_item")
            item_pred = None
            

        category_fixed = original_category
        category_changed = False

        # ---------------- 2) RagFlow spec patterns using "item_pred + category_fixed" ----------------
        spec_query_parts = []
        if category_fixed:
            spec_query_parts.append(str(category_fixed))
        if item_pred:
            spec_query_parts.append(str(item_pred))
        spec_query = " ".join(spec_query_parts).strip()

        if spec_query:
            spec_patterns_data = await self.rag.get_spec_patterns_by_query(spec_query)
        else:
            spec_patterns_data = await self.rag.get_spec_patterns_by_query(str(description))

        # Sort by similarity descending
        spec_patterns_data.sort(key=lambda x: x.get("similarity", 0) or 0, reverse=True)

        # Format with similarity score
        spec_patterns_list = []
        for p in spec_patterns_data:
            sim = p.get("similarity", 0) or 0
            text = p.get("spec", "")
            spec_patterns_list.append(f"[Similarity: {sim:.4f}] {text}")

        spec_patterns_text = "\n".join(spec_patterns_list)


        # ---------------- 3) FIX SPEC using fix_spec prompt ----------------
        try:
            fix_result = await asyncio.wait_for(
                self.llm.afix_spec(
                    prompts["fix_spec"],
                    description=description,
                    spec_pred=original_spec_pred,
                    item_pred=item_pred or "",
                    category_fixed=category_fixed or "",
                    spec_patterns=spec_patterns_text,
                ),
                timeout=120,
            )
            spec_after_fix = fix_result.spec_pred_fixed
            # print(spec_after_fix)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] fix_spec at row with description={description[:80]}")
            spec_after_fix = original_spec_pred
        except Exception as e:
            print(f"[LLM] fix_spec structured error: {e!r}")
            spec_after_fix = original_spec_pred

        
        # ---------------- 4) REMOVE Multiple 'item' keys usiing remove_multi_items prompt ----------------
        try:
            remove_result = await asyncio.wait_for(
                self.llm.aremove_multi_items(
                    prompts["remove_multi_items"],
                    description=description,
                    spec_pred_fixed=spec_after_fix,
                    category_fixed=category_fixed or "",
                ),
                timeout=60
            )
            spec_after_remove_items = remove_result.spec_pred_remove_items
            # print("=== BEFORE VALIDATE ===", spec_after_fix)
            # print("=== AFTER VALIDATE ====", spec_after_remove_items)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] remove_multi_items at row with description={description[:80]}")
            spec_after_remove_items = spec_after_fix
        except Exception as e:
            print(f"[LLM] validate_spec error: {e!r}")
            spec_after_remove_items = spec_after_fix


        # ---------------- 5) VALIDATE spec_pred_fixed against description & clean hallucinated values --------
        try:
            validate_result = await asyncio.wait_for(
                self.llm.avalidate_spec(
                    prompts["validate_spec"],
                    description=description,
                    spec_pred_remove_items=spec_after_remove_items,
                ),
                timeout=60,
            )
            final_spec = validate_result.spec_pred_fixed_validated
            # print("=== BEFORE VALIDATE ===", spec_after_remove_items)
            # print("=== AFTER VALIDATE ====", final_spec)

            # ---------------- 5.1) ALIGN keys with original spec_pred to remove extra keys --------
            final_spec = align_spec_keys(original_spec_pred, final_spec)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] validate_spec at row with description={description[:80]}")
            final_spec = spec_after_remove_items
        except Exception as e:
            print(f"[LLM] validate_spec error: {e!r}")
            final_spec = spec_after_remove_items

        # ---------------- Detect change ----------------
        norm_original_spec = self._normalize_for_compare(original_spec_pred)
        norm_fixed_spec = self._normalize_for_compare(final_spec)
        spec_changed = norm_fixed_spec != "" and (norm_fixed_spec != norm_original_spec)

        # ---------------- 6) Extract item_extracted from FIXED spec ----------------
        if final_spec:
            item_extracted = extract_item(final_spec)
        else:
            item_extracted = None

        return {
            "item_pred": item_pred,
            "item_extracted": item_extracted,
            "spec_pred_fixed": final_spec,
            "category_fixed": category_fixed,
            "spec_changed": spec_changed,
            "category_changed": category_changed,
        }

import csv
import asyncio
from typing import List, Any, Dict
from ragflow_sdk import RAGFlow

from app.utils.config import (
    RAGFLOW_URL,
    RAGFLOW_API_KEY,
    RAGFLOW_PO_DATASET_IDS,
    TOP_K,
)


class RagFlowService:
    def __init__(self):
        try:
            self.rag_client = RAGFlow(
                api_key=RAGFLOW_API_KEY,
                base_url=RAGFLOW_URL,
            )
            print("[RAGFLOW] Client initialized successfully.")
        except Exception as e:
            print(f"[RAGFLOW] Could not initialize RAGFlow client: {e}. Falling back to empty results.")
            self.rag_client = None



    async def _retrieve(self, question: str, top_k: int = None) -> List[Any]:
        if not self.rag_client:
            return []

        if top_k is None:
            top_k = TOP_K

        loop = asyncio.get_running_loop()

        def _call():
            # sync call into SDK
            return self.rag_client.retrieve(
                dataset_ids=RAGFLOW_PO_DATASET_IDS,
                question=question,
                top_k=top_k,
            )

        try:
            # hard timeout so no single RAG call can hang the whole pipeline
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=45,  # seconds
            )
            return list(result or [])
        except asyncio.TimeoutError:
            print(f"[RAGFLOW TIMEOUT] question='{question}' (top_k={top_k})")
            return []
        except Exception as e:
            print(f"[RAGFLOW] Error during retrieve (question='{question}'): {e}")
            return []

    # --------- helpers for flexible column names ---------

    @staticmethod
    def _parse_chunk_text(text: str) -> Dict[str, Any]:
        """
        Parse RagFlow chunk text of the form:
        header1,header2,...,headerN:value1,value2,...,valueN

        Handles quoted headers/values and commas properly using csv.reader.
        Returns dict mapping header -> value (both as strings).
        """
        if not text or ":" not in text:
            return {}

        header_part, value_part = text.split(":", 1)

        # Use csv.reader to correctly parse quotes and commas
        header_reader = csv.reader([header_part])
        value_reader = csv.reader([value_part])

        try:
            headers = next(header_reader)
            values = next(value_reader)
        except StopIteration:
            return {}

        # strip whitespace around headers/values
        headers = [h.strip() for h in headers]
        values = [v.strip() for v in values]

        # zip into dict
        row = {}
        for h, v in zip(headers, values):
            row[h] = v

        return row

    @staticmethod
    def _get_from_mapping(mapping, candidate_keys):
        """Try multiple key variants against a dict-like mapping."""
        if not mapping:
            return None
        for k in candidate_keys:
            if k in mapping:
                return mapping[k]
        # normalized (lowercase, no spaces/quotes/underscores)
        norm_map = {
            "".join(ch for ch in key.lower() if ch.isalnum()): v
            for key, v in mapping.items()
        }
        for k in candidate_keys:
            norm_k = "".join(ch for ch in k.lower() if ch.isalnum())
            if norm_k in norm_map:
                return norm_map[norm_k]
        return None

    @classmethod
    def _get_field(cls, rec: Any, candidate_keys: List[str]):
        """
        Try to get a field from:
        - top-level dict
        - rec.metadata (if object-like)
        """
        if isinstance(rec, dict):
            val = cls._get_from_mapping(rec, candidate_keys)
            if val is not None:
                return val
            meta = rec.get("metadata") or {}
            return cls._get_from_mapping(meta, candidate_keys)

        # object-like result
        meta = getattr(rec, "metadata", {}) or {}
        # direct attributes first
        for k in candidate_keys:
            if hasattr(rec, k):
                return getattr(rec, k)
        # metadata attributes / keys 
        val = cls._get_from_mapping(meta, candidate_keys)
        if val is not None:
            return val

        return None

    # --------- what asked about: Subgroup / ItemDescription / spec ---------

    @classmethod
    def _extract_spec_patterns(cls, retrieval_results: List[Any]) -> List[str]:
        """
        Extract spec pattern text from results.
        - First, try structured fields: spec / "spec" / etc.
        - If not found, and record looks like the CSV-chunk text  parse it.
        """
        patterns = []

        for rec in retrieval_results:
            spec = None

            # 1) Try structured dict/object fields first
            if isinstance(rec, dict):
                # top-level
                spec = rec.get("spec") or rec.get("Spec")
                # or from metadata
                if spec is None and "metadata" in rec:
                    meta = rec.get("metadata") or {}
                    spec = meta.get("spec") or meta.get("Spec")
            else:
                meta = getattr(rec, "metadata", {}) or {}
                spec = (
                    getattr(rec, "spec", None)
                    or getattr(rec, "Spec", None)
                    or meta.get("spec")
                    or meta.get("Spec")
                )

            # 2) If still None, try parsing 'content' / 'text' as chunk CSV
            if spec is None:
                if isinstance(rec, dict):
                    text = rec.get("content") or rec.get("text") or ""
                else:
                    text = getattr(rec, "content", "") or getattr(rec, "text", "") or ""

                row_dict = cls._parse_chunk_text(text)
                # spec column name can be spec or "spec"
                spec = row_dict.get("spec") or row_dict.get('"spec"')

            if spec:
                patterns.append(str(spec))

        return patterns

    @classmethod
    def _extract_categories(cls, retrieval_results: List[Any]) -> List[str]:
        """
        Extract subgroup/category from results.
        Supports:
        - subgroup / Subgroup
        - "subgroup"
        and from parsed CSV text.
        """
        categories = []

        for rec in retrieval_results:
            subgroup = None

            # 1) Structured
            if isinstance(rec, dict):
                subgroup = rec.get("subgroup") or rec.get("Subgroup")
                if subgroup is None and "metadata" in rec:
                    meta = rec.get("metadata") or {}
                    subgroup = meta.get("subgroup") or meta.get("Subgroup")
            else:
                meta = getattr(rec, "metadata", {}) or {}
                subgroup = (
                    getattr(rec, "subgroup", None)
                    or getattr(rec, "Subgroup", None)
                    or meta.get("subgroup")
                    or meta.get("Subgroup")
                )

            # 2) Parsed from chunk text
            if subgroup is None:
                if isinstance(rec, dict):
                    text = rec.get("content") or rec.get("text") or ""
                else:
                    text = getattr(rec, "content", "") or getattr(rec, "text", "") or ""

                row_dict = cls._parse_chunk_text(text)
                subgroup = (
                    row_dict.get("Subgroup")
                    or row_dict.get("subgroup")
                    or row_dict.get('"subgroup"')
                )

            if subgroup:
                categories.append(str(subgroup))

        return categories
    
    @classmethod
    def _extract_item_descriptions(cls, retrieval_results: List[Any]) -> List[str]:
        """
        Optional: extract item description.
        Supports:
        - ItemDescription
        - item_description
        - from parsed chunk text.
        """
        items = []

        for rec in retrieval_results:
            item_desc = None

            if isinstance(rec, dict):
                item_desc = rec.get("ItemDescription") or rec.get("item_description")
                if item_desc is None and "metadata" in rec:
                    meta = rec.get("metadata") or {}
                    item_desc = meta.get("ItemDescription") or meta.get("item_description")
            else:
                meta = getattr(rec, "metadata", {}) or {}
                item_desc = (
                    getattr(rec, "ItemDescription", None)
                    or getattr(rec, "item_description", None)
                    or meta.get("ItemDescription")
                    or meta.get("item_description")
                )

            if item_desc is None:
                if isinstance(rec, dict):
                    text = rec.get("content") or rec.get("text") or ""
                else:
                    text = getattr(rec, "content", "") or getattr(rec, "text", "") or ""

                row_dict = cls._parse_chunk_text(text)
                item_desc = (
                    row_dict.get("ItemDescription")
                    or row_dict.get("item_description")
                    or row_dict.get('"item_description"')
                )

            if item_desc:
                items.append(str(item_desc))

        return items

    

    async def get_spec_patterns_by_description(self, description: str):
        """Use description as query to get spec patterns."""
        results = await self._retrieve(description or "")
        return self._extract_spec_patterns(results)

    async def get_spec_patterns_by_query(self, query: str):
        """Generic: use any query (item_pred, item_pred + category, etc.) to get spec patterns."""
        results = await self._retrieve(query or "")
        return self._extract_spec_patterns(results)

    async def get_categories_by_query(self, query: str):
        """Use any query (e.g. item_extracted, item_pred, or description) to get categories."""
        results = await self._retrieve(query or "")
        return self._extract_categories(results)


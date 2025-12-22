from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.utils.config import MODEL_URL, MODEL_NAME, MODEL_TEMPERATURE, MODEL_API_KEY


from app.utils.spec_models import (
    PredictItemResult,
    FixCategoryResult,
    ValidateSpecResult,
    FixSpecResult,
    RemoveMultipleItem,
)


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            streaming=False,
            temperature=MODEL_TEMPERATURE,
            model=MODEL_NAME,
            openai_api_base=MODEL_URL,
            openai_api_key=MODEL_API_KEY
        )

    async def ainvoke_prompt(self, prompt_template: str, **kwargs):
        """
        Generic unstructured LLM call used by _safe_llm_call.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        return await chain.ainvoke(kwargs)


    # ------------------------------------------------------
    # 1) Predict Item
    # ------------------------------------------------------

    async def apredict_item(self, prompt_template: str, *, description: str) -> PredictItemResult:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm.with_structured_output(PredictItemResult)

        return await chain.ainvoke(
            {
                "description": description,
            }
        )

    # ------------------------------------------------------
    # 2) Fix Category
    # ------------------------------------------------------

    async def afix_category(
        self,
        prompt_template: str,
        *,
        description: str,
        category: str,
        item: str,
        rag_categories: str,
    ) -> FixCategoryResult:

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm.with_structured_output(FixCategoryResult)

        return await chain.ainvoke(
            {
                "description": description,
                "category": category,
                "item": item,
                "rag_categories": rag_categories,
            }
        )

    # ------------------------------------------------------
    # 3) Fix Spec
    # ------------------------------------------------------

    async def afix_spec(
        self,
        prompt_template: str,
        *,
        description: str,
        spec_pred: str,
        item_pred: str,
        category_fixed: str,
        spec_patterns: str,
    ) -> FixSpecResult:
        
        system_message = """You are a product specification expert who corrects and standardizes spec data.

        CORE PRINCIPLES:
        1. Extract values ONLY from the description - never hallucinate or copy from examples
        2. Follow consistent naming patterns from spec_patterns while extracting real values from description
        3. Distinguish between model codes and measurements with units
        4. Use "-" for missing information, never guess

        CRITICAL RULES:
        - Values come from description ONLY (except "item" key which follows pattern style)
        - Measurements (50kg, 100l, 220v) go to measurement keys, NOT model key
        - Model codes are alphanumeric identifiers, NOT numbers with units
        - Never add category, subgroup, or product type keys
        - Return exact JSON format requested, no extra text

        Your goal: Create clean, accurate specs that match the description facts while following consistent naming patterns from examples.
        """

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", prompt_template),
        ]
        )
        chain = prompt | self.llm.with_structured_output(FixSpecResult)

        return await chain.ainvoke(
            {
                "description": description,
                "spec_pred": spec_pred,
                "item_pred": item_pred,
                "category_fixed": category_fixed,
                "spec_patterns": spec_patterns,
            }
        )


    # ------------------------------------------------------
    # 4) REMOVE Multiple 'item' keys
    # ------------------------------------------------------
    async def aremove_multi_items(
        self,
        prompt_template: str,
        *,
        spec_pred_fixed: str,
        description: str,
        category_fixed: str
    ) -> RemoveMultipleItem:
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm.with_structured_output(RemoveMultipleItem)

        return await chain.ainvoke(
            {
                "description": description,
                "spec_pred_fixed": spec_pred_fixed,
                "category_fixed": category_fixed,
            }
        )
        

    # ------------------------------------------------------
    # 5) Validate spec_pred_remove_items
    # ------------------------------------------------------
    async def avalidate_spec(
        self,
        prompt_template: str,
        *,
        description: str,
        spec_pred_remove_items: str,
    ) -> ValidateSpecResult:
        """
        Validator:
        - Check each key/value in spec_pred_remove_items against description.
        - If value not in description, try to refind from description.
        - If still unknown, replace with "-".
        """

        system_message = """You are a fact-checker correcting hallucinated data.

        Another AI filled in values from a description. Some values may be WRONG (hallucinated).

        ## Description
        {description}

        ## Current Spec Data (may contain hallucinations)
        {spec_pred_remove_items}

        Your job: Check each value against the description. Fix wrong values.

        ABSOLUTE RULE: Return the EXACT SAME list of key-value pairs. Same keys, same order, same count.
        NEVER add keys. NEVER remove keys. ONLY fix values that are hallucinated.

        ABSOLUTE CONSTRAINTS (violating = failure):
        - Same number of keys in output as spec_pred
        - Same key names and order as input
        - Do NOT add keys
        - Do NOT remove keys
        - Do NOT touch "-" values (already marked unknown)
        - Do NOT touch "item" key

        If unsure about a value, use "-" instead of guessing.
        """

        llm_validate_spec = ChatOpenAI(
            streaming=False,
            temperature=0,            # <-- ONLY HERE, not global
            model=MODEL_NAME,
            openai_api_base=MODEL_URL,
            openai_api_key=MODEL_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", prompt_template),
        ]
        )
        chain = prompt | llm_validate_spec.with_structured_output(ValidateSpecResult)


        return await chain.ainvoke(
            {
                "description": description,
                "spec_pred_remove_items": spec_pred_remove_items,
            }
        )

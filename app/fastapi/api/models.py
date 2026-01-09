from pydantic import BaseModel, Field
from typing import Optional, List


class RowIn(BaseModel):
    description: str
    spec_pred: Optional[str] = None
    category: Optional[str] = None


class FixRowOut(BaseModel):
    item_pred: Optional[str]
    item_extracted: Optional[str]
    spec_pred_fixed: Optional[str]
    category_fixed: Optional[str]
    spec_changed: bool
    category_changed: bool


class PredictItemIn(BaseModel):
    description: str


class PredictItemOut(BaseModel):
    item_pred: str


class FixCategoryIn(BaseModel):
    description: str
    category: str
    item: str = ""
    rag_categories: str = ""


class FixCategoryOut(BaseModel):
    category_fixed: str


class FixSpecIn(BaseModel):
    description: str
    spec_pred: str = ""
    item_pred: str = ""
    category_fixed: str = ""
    spec_patterns: str = ""


class FixSpecOut(BaseModel):
    spec_pred_fixed: str


class RemoveMultiItemsIn(BaseModel):
    description: str
    spec_pred_fixed: str
    category_fixed: str = ""


class RemoveMultiItemsOut(BaseModel):
    spec_pred_remove_items: str


class ValidateSpecIn(BaseModel):
    description: str
    spec_pred_remove_items: str


class ValidateSpecOut(BaseModel):
    spec_pred_fixed_validated: str



class PostProcessRowOut(BaseModel):
    description: str
    category: Optional[str]
    spec_pred: Optional[str]


class BatchFixIn(BaseModel):
    rows: List[RowIn]
    post_process: bool = True

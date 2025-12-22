from pydantic import BaseModel, Field, field_validator
from app.utils.spec_parser import fix_spec_format


class PredictItemResult(BaseModel):
    item_pred: str = Field(..., description="Predicted product or service item name")


class FixCategoryResult(BaseModel):
    category_fixed: str = Field(..., description="Final corrected category/subgroup")


class FixSpecResult(BaseModel):
    spec_pred_fixed: str = Field(..., description="Spec after fix_spec")


class RemoveMultipleItem(BaseModel):
    spec_pred_remove_items: str = Field(..., description="Spec with only one item key")


class ValidateSpecResult(BaseModel):
    spec_pred_fixed_validated: str = Field(..., description="Validated final spec")
    @field_validator("spec_pred_fixed_validated")
    @classmethod
    def normalize_spec(cls, v: str) -> str:
        return fix_spec_format(v or "")

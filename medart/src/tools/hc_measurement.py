from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.services.csm_inference import measure_single_image


class MeasureFetalHeadCircumferenceInput(BaseModel):
    image_path: str = Field(description="Input: Absolute path to the local ultrasound image, required.")
    pixel_size_mm: float = Field(
        description="Input: Millimeters per pixel, required, unit: mm/pixel."
    )
    case_id: str | None = Field(
        default=None,
        description="Input: Sample or case identifier, optional.",
    )



@tool(args_schema=MeasureFetalHeadCircumferenceInput)
def measure_fetal_head_circumference(
    image_path: str,
    pixel_size_mm: float,
    case_id: str | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Measure fetal head circumference from a fetal head ultrasound image.

    Inputs:
    - image_path: Absolute path to the local ultrasound image, required
    - pixel_size_mm: Millimeters per pixel, required, unit: mm/pixel
    - case_id: Sample or case identifier, optional

    Outputs:
    - hc_mm: Fetal head circumference in millimeters
    - warnings: List of non-fatal warnings
    """
    result = measure_single_image(
        image_path=image_path,
        pixel_size_mm=pixel_size_mm,
        case_id=case_id,
        return_debug=return_debug,
    )
    return {
        "hc_mm": result.get("hc_mm"),
        "warnings": result.get("warnings", []),
    }

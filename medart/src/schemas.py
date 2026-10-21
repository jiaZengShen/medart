from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MeasurementToolResult:
    status: str
    image_path: str
    pixel_size_mm: float | None
    hc_px: float | None = None
    hc_mm: float | None = None
    ellipse: dict[str, float] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    case_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPrediction:
    sample_id: str
    claim: str
    image_path: str
    true_label: int
    pred_label: int
    generated_text: str
    tool_called: bool
    tool_result: dict[str, Any]
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

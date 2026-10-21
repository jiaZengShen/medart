import importlib.util
import json
import sys
import traceback
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

#from src.config import CSM_ROOT, DEFAULT_CSM_WEIGHTS
from src.schemas import MeasurementToolResult


def _to_float(value):
    if value is None:
        return None
    return float(value)


def _read_grayscale_image(image_file: Path):
    buffer = np.fromfile(str(image_file), dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


@lru_cache(maxsize=1)
def _load_csm_module():
    module_path =  Path(r"./src/services/csm/modules.py")
    spec = importlib.util.spec_from_file_location("measure7_csm_modules", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CSMInferenceService:
    def __init__(self, weights_path: Path = Path(r"./src/services/csm/test_model.pth") ):
        self.weights_path = Path(weights_path)
        self._model = None
        self._device = None
        self._module = _load_csm_module()
        self._model_lock = threading.Lock()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def _load_model(self, device: str):
        resolved_device = self._resolve_device(device)
        with self._model_lock:
            if self._model is not None and self._device == resolved_device:
                return self._model, resolved_device

            model = self._module.CSM()
            state_dict = torch.load(self.weights_path, map_location=resolved_device)
            model.load_state_dict(state_dict)
            model.to(resolved_device)
            model.eval()
            self._model = model
            self._device = resolved_device
            return model, resolved_device

    def measure_single_image(
        self,
        image_path: str,
        pixel_size_mm: float | None = None,
        device: str = "cuda",
        return_debug: bool = False,
        case_id: str | None = None,
    ) -> MeasurementToolResult:
        image_file = Path(image_path)
        if not image_file.exists():
            return MeasurementToolResult(
                status="invalid_image",
                image_path=str(image_file),
                pixel_size_mm=_to_float(pixel_size_mm),
                warnings=["image file does not exist"],
                case_id=case_id,
            )

        raw_image = _read_grayscale_image(image_file)
        if raw_image is None:
            return MeasurementToolResult(
                status="invalid_image",
                image_path=str(image_file),
                pixel_size_mm=_to_float(pixel_size_mm),
                warnings=["failed to decode image"],
                case_id=case_id,
            )

        model, resolved_device = self._load_model(device)
        cropped = self._module.img_crop(raw_image)
        input_img = cv2.resize(cropped, (192, 128), interpolation=cv2.INTER_AREA)
        normalized = (input_img.astype("float32") / 255.0)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(resolved_device)

        with torch.no_grad():
            _, _, predict = model(tensor)

        mask = predict[0, 0, :, :].detach().cpu().numpy()
        mask = np.round(mask) * 255
        mask = mask.astype("uint8")
        mask_area_px = int(np.count_nonzero(mask))
        if mask_area_px == 0:
            return MeasurementToolResult(
                status="no_head_found",
                image_path=str(image_file),
                pixel_size_mm=_to_float(pixel_size_mm),
                preprocessing={
                    "original_shape": list(raw_image.shape),
                    "cropped_shape": list(cropped.shape),
                    "resized_shape": list(input_img.shape),
                    "device": resolved_device,
                },
                quality={"mask_area_px": 0, "edge_points": 0},
                warnings=["predicted mask is empty"],
                case_id=case_id,
            )

        edge = self._module.mcc_edge(mask)
        edge_points = int(np.count_nonzero(edge))
        if edge_points < 5:
            return MeasurementToolResult(
                status="fit_failed",
                image_path=str(image_file),
                pixel_size_mm=_to_float(pixel_size_mm),
                preprocessing={
                    "original_shape": list(raw_image.shape),
                    "cropped_shape": list(cropped.shape),
                    "resized_shape": list(input_img.shape),
                    "device": resolved_device,
                },
                quality={"mask_area_px": mask_area_px, "edge_points": edge_points},
                warnings=["not enough edge points for ellipse fitting"],
                case_id=case_id,
            )

        try:
            xc, yc, theta, a, b = self._module.ellip_fit(edge)
        except Exception as exc:
            traceback.print_exc()
            return MeasurementToolResult(
                status="fit_failed",
                image_path=str(image_file),
                pixel_size_mm=_to_float(pixel_size_mm),
                preprocessing={
                    "original_shape": list(raw_image.shape),
                    "cropped_shape": list(cropped.shape),
                    "resized_shape": list(input_img.shape),
                    "device": resolved_device,
                },
                quality={"mask_area_px": mask_area_px, "edge_points": edge_points},
                warnings=[f"ellipse fitting failed: {exc}"],
                case_id=case_id,
            )

        upsample = 16
        xc = (float(xc) + 0.5) * upsample - 0.5
        yc = (float(yc) + 0.5) * upsample - 0.5
        a = float(a) * upsample
        b = float(b) * upsample
        theta = float(theta)
        hc_px = float(2 * np.pi * b + 4 * (a - b))
        hc_mm = float(hc_px * pixel_size_mm) if pixel_size_mm is not None else None

        warnings = []
        if pixel_size_mm is None:
            warnings.append("pixel_size_mm is missing, hc_mm is unavailable")

        quality = {
            "mask_area_px": mask_area_px,
            "edge_points": edge_points,
        }
        if return_debug:
            quality["mask_sum"] = int(mask.sum())
            quality["edge_sum"] = int(edge.sum())

        status = "ok" if pixel_size_mm is not None else "missing_pixel_size"
        return MeasurementToolResult(
            status=status,
            image_path=str(image_file),
            pixel_size_mm=_to_float(pixel_size_mm),
            hc_px=hc_px,
            hc_mm=hc_mm,
            ellipse={
                "center_x": xc,
                "center_y": yc,
                "semi_axes_a": a,
                "semi_axes_b": b,
                "angle_rad": theta,
            },
            preprocessing={
                "original_shape": list(raw_image.shape),
                "cropped_shape": list(cropped.shape),
                "resized_shape": list(input_img.shape),
                "device": resolved_device,
            },
            quality=quality,
            warnings=warnings,
            case_id=case_id,
        )


@lru_cache(maxsize=1)
def get_default_service() -> CSMInferenceService:
    return CSMInferenceService()


def measure_single_image(image_path: str, pixel_size_mm: float | None = None, device: str = "auto", return_debug: bool = False, case_id: str | None = None) -> dict:
    return get_default_service().measure_single_image(
        image_path=image_path,
        pixel_size_mm=pixel_size_mm,
        device=device,
        return_debug=return_debug,
        case_id=case_id,
    ).to_dict()


def format_tool_evidence(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)

"""
Side Analysis Module - Analyze segmentation mask distribution for side verification.

This module provides functions to:
1. Load segmentation masks from PNG files
2. Analyze mask pixel distribution across left/right halves of the image
3. Extract side keywords from claims
4. Extract segmentation targets from claims
"""

from __future__ import annotations

import re
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


def load_mask(mask_path: str | Path) -> Optional[np.ndarray]:
    """
    Load a segmentation mask from a PNG file.

    Args:
        mask_path: Path to the PNG mask file

    Returns:
        Binary numpy array (H, W) where 1 = mask region, 0 = background
        Returns None if loading fails
    """
    try:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            return None

        img = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask_array = np.array(img)

        # Convert to binary (0 or 1) - threshold at 128
        binary_mask = (mask_array > 128).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"Error loading mask: {e}")
        return None


def analyze_side_distribution(mask: np.ndarray) -> Dict[str, Any]:
    """
    Analyze mask pixel distribution across left/right halves of the image.

    Args:
        mask: Binary numpy array (H, W) where 1 = mask region, 0 = background

    Returns:
        Dictionary with analysis results:
        {
            "side": "left" | "right" | "unknown",
            "confidence": float,  # Ratio of pixels on the dominant side
            "left_ratio": float,  # Ratio of mask pixels on left side
            "right_ratio": float,  # Ratio of mask pixels on right side
            "total_pixels": int,  # Total mask pixels
            "left_pixels": int,  # Mask pixels on left side
            "right_pixels": int,  # Mask pixels on right side
            "image_width": int,  # Image width
            "image_height": int,  # Image height
        }
    """
    if mask is None or len(mask.shape) != 2:
        return {
            "side": "unknown",
            "confidence": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "total_pixels": 0,
            "left_pixels": 0,
            "right_pixels": 0,
            "image_width": 0,
            "image_height": 0,
        }

    height, width = mask.shape
    mid_x = width // 2

    # Count pixels on each side
    left_pixels = int(np.sum(mask[:, :mid_x]))
    right_pixels = int(np.sum(mask[:, mid_x:]))
    total_pixels = left_pixels + right_pixels

    if total_pixels == 0:
        return {
            "side": "unknown",
            "confidence": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "total_pixels": 0,
            "left_pixels": 0,
            "right_pixels": 0,
            "image_width": width,
            "image_height": height,
        }

    left_ratio = left_pixels / total_pixels
    right_ratio = right_pixels / total_pixels

    # Determine side
    if left_ratio > right_ratio:
        side = "left"
        confidence = left_ratio
    elif right_ratio > left_ratio:
        side = "right"
        confidence = right_ratio
    else:
        side = "unknown"
        confidence = 0.5

    return {
        "side": side,
        "confidence": confidence,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "total_pixels": total_pixels,
        "left_pixels": left_pixels,
        "right_pixels": right_pixels,
        "image_width": width,
        "image_height": height,
    }


def extract_claim_side(claim: str) -> Optional[str]:
    """
    Extract the mentioned side from a claim.

    Args:
        claim: The claim text to analyze

    Returns:
        "left", "right", or None if no side is mentioned
    """
    claim_lower = claim.lower()

    # English keywords
    left_patterns = [r'\bleft\b', r'\bL\b']
    right_patterns = [r'\bright\b', r'\bR\b']

    # Chinese keywords
    left_patterns += [r'左', r'左侧']
    right_patterns += [r'右', r'右侧']

    has_left = any(re.search(pattern, claim, re.IGNORECASE) for pattern in left_patterns)
    has_right = any(re.search(pattern, claim, re.IGNORECASE) for pattern in right_patterns)

    if has_left and not has_right:
        return "left"
    elif has_right and not has_left:
        return "right"
    elif has_left and has_right:
        # Both mentioned - need more sophisticated handling
        # For now, return the one that appears first
        left_match = re.search(r'\bleft\b|左|左侧', claim, re.IGNORECASE)
        right_match = re.search(r'\bright\b|右|右侧', claim, re.IGNORECASE)
        if left_match and right_match:
            return "left" if left_match.start() < right_match.start() else "right"
        elif left_match:
            return "left"
        elif right_match:
            return "right"

    return None


def extract_segmentation_target_with_llm(claim: str, llm) -> str:
    """
    Extract a segmentation target description from a claim using LLM.

    Args:
        claim: The claim text
        llm: An LLM instance with an invoke method that takes messages

    Returns:
        A prompt suitable for BiomedParse segmentation
    """
    extraction_prompt = """You are a medical image analysis assistant. Your task is to extract what anatomical structure or finding needs to be segmented from the following claim.

IMPORTANT RULES:
1. DO NOT include any side information (left, right, left side, right side, 左, 右, 左侧, 右侧) in your response
2. Focus only on the anatomical structure or the finding itself
3. Be concise - respond with just the target description in 5-20 words
4. Do NOT include any reasoning, just the target description

Claim to analyze:
{claim}

What needs to be segmented in the image? (Do NOT mention left/right)""".format(claim=claim)

    try:
        response = llm.invoke([{"role": "user", "content": extraction_prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Clean up the response
        response_text = response_text.strip()

        # Remove any remaining side references as a safeguard
        side_patterns = [
            r'\bleft\b', r'\bright\b', r'\bL\b', r'\bR\b',
            r'左', r'右', r'左侧', r'右侧'
        ]
        for pattern in side_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)

        response_text = re.sub(r'\s+', ' ', response_text).strip()

        if response_text:
            return response_text

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整调用栈
        print(f"LLM extraction failed: {e}, falling back to simple method")

    # Fallback if LLM fails
    return extract_segmentation_target_simple(claim)


def extract_segmentation_target_simple(claim: str) -> str:
    """
    Simple fallback method to extract segmentation target (without LLM).
    """
    claim_lower = claim.lower()

    # Common anatomical structures (extend this list as needed)
    anatomical_keywords = [
        # Brain
        "brain", "cerebrum", "cerebellum", "hemisphere", "lobe",
        # Thorax
        "lung", "heart", "pleura", "mediastinum",
        # Abdomen
        "liver", "spleen", "kidney", "gallbladder", "pancreas",
        # Other
        "organ", "tissue", "lesion", "mass", "tumor", "nodule",
        "finding", "abnormality", "calcification",
        # Fetal/ultrasound
        "fetus", "fetal", "head", "abdomen",
    ]

    # Look for anatomical structures
    found_structures = []
    for keyword in anatomical_keywords:
        if keyword in claim_lower:
            found_structures.append(keyword)

    if found_structures:
        # Use the first found structure
        return f"the {found_structures[0]} mentioned in the claim"

    # Generic fallbacks
    if "abnorm" in claim_lower or "lesion" in claim_lower:
        return "the abnormal region or lesion mentioned in the claim"
    elif "tumor" in claim_lower or "mass" in claim_lower:
        return "the tumor or mass mentioned in the claim"

    # Ultimate fallback
    return "the finding or anatomical structure mentioned in the claim"


# Keep the original name for compatibility, but it now uses the simple method
def extract_segmentation_target(claim: str) -> str:
    """
    Extract segmentation target (simple fallback method).
    Use extract_segmentation_target_with_llm when an LLM is available.
    """
    return extract_segmentation_target_simple(claim)


def format_side_analysis_for_llm(analysis: Dict[str, Any], claim_side: Optional[str] = None) -> str:
    """
    Format side analysis results into a human-readable string for LLM consumption.

    Args:
        analysis: The analysis result from analyze_side_distribution()
        claim_side: The side mentioned in the claim (optional)

    Returns:
        Formatted string
    """
    lines = []
    lines.append("### BiomedParse Side Analysis Result")
    lines.append("")

    if analysis["total_pixels"] == 0:
        lines.append("⚠️ No significant segmentation region detected, cannot perform side analysis.")
        return "\n".join(lines)

    # Side determination
    side_display = {
        "left": "Left",
        "right": "Right",
        "unknown": "Unknown"
    }
    lines.append(f"- **Detected Primary Side**: {side_display.get(analysis['side'], analysis['side'])}")
    lines.append(f"- **Confidence**: {analysis['confidence']:.1%}")
    lines.append("")

    # Pixel distribution
    lines.append("- **Pixel Distribution**:")
    lines.append(f"  - Left: {analysis['left_pixels']} pixels ({analysis['left_ratio']:.1%})")
    lines.append(f"  - Right: {analysis['right_pixels']} pixels ({analysis['right_ratio']:.1%})")
    lines.append(f"  - Total: {analysis['total_pixels']} pixels")
    lines.append("")

    # Comparison with claim side if available
    if claim_side:
        claim_side_display = "Left" if claim_side == "left" else "Right"
        lines.append(f"- **Side Mentioned in Claim**: {claim_side_display}")

        if analysis["side"] != "unknown" and claim_side != analysis["side"]:
            lines.append("")
            lines.append("⚠️ **WARNING**: The primary side detected by BiomedParse does NOT match the side mentioned in the claim!")
            lines.append(f"   - Detected: {side_display.get(analysis['side'])}")
            lines.append(f"   - Claim: {claim_side_display}")

    return "\n".join(lines)


def analyze_mask_from_path(mask_path: str | Path, claim: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function: Load mask and analyze in one step.

    Args:
        mask_path: Path to the mask file
        claim: Optional claim text for side extraction

    Returns:
        Combined analysis result
    """
    mask = load_mask(mask_path)
    analysis = analyze_side_distribution(mask)

    if claim:
        analysis["claim_side"] = extract_claim_side(claim)

    return analysis

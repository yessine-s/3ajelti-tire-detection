"""Plain-language guidance associated with image model labels.

This module provides general next steps only; the model cannot assess tire safety.
"""

TIRE_INFO = {
    "Bulge": {
        "description": "A visible sidewall bulge can indicate internal tire damage. A photo classifier cannot confirm the cause or severity.",
        "model_signal": "Possible bulge detected",
        "advice": "If a bulge is visible, stop using the tire and have it assessed by a qualified tire professional.",
        "color": "#FF4444",
        "icon": "🚨",
    },
    "Cracks": {
        "description": "The model detected a possible crack or other visible damage. A professional inspection is needed to assess its extent.",
        "model_signal": "Possible damage detected",
        "advice": "If cracks or other damage are visible, stop using the tire and have it assessed by a qualified tire professional.",
        "color": "#FFA500",
        "icon": "⚠️",
    },
    "Non-defective": {
        "description": "No visible defect was detected in this image. This does not confirm that the tire is safe or free of hidden damage.",
        "model_signal": "No defect detected in this image",
        "advice": "Continue regular tire inspections and consult a qualified professional if you notice damage or a change in performance.",
        "color": "#4CAF50",
        "icon": "✅",
    },
}


def get_tire_info(defect_class: str) -> dict:
    """Return general, non-quantified guidance for a model class."""
    return TIRE_INFO.get(defect_class, TIRE_INFO["Non-defective"])
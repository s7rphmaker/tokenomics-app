import json
import math
from typing import Any

from fastapi.responses import JSONResponse


class SafeJSONResponse(JSONResponse):
    """JSONResponse that converts NaN/Infinity to null for JSON compliance."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            default=self._default,
        ).encode("utf-8")

    @staticmethod
    def _default(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def sanitize_floats(obj):
    """Recursively replace NaN/Infinity with None in nested structures."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj

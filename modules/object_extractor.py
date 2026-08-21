from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

SYNONYMS = {
    "person": ["người", "con người", "đàn ông", "phụ nữ", "trẻ em", "em bé", "cô gái", "chàng trai", "man", "woman"],
    "car": ["xe hơi", "ô tô", "oto", "xe con", "car"],
    "motorcycle": ["xe máy", "mô tô", "motorbike", "motorcycle"],
    "bicycle": ["xe đạp", "bicycle", "bike"],
    "bus": ["xe buýt", "bus"],
    "truck": ["xe tải", "truck"],
    "traffic light": ["đèn giao thông", "đèn đỏ", "traffic light"],
    "dog": ["chó", "dog"],
    "cat": ["mèo", "cat"],
    "bird": ["chim", "bird"],
    "chair": ["ghế", "chair"],
    "couch": ["sofa", "ghế sofa", "couch"],
    "dining table": ["bàn", "bàn ăn", "table"],
    "laptop": ["laptop", "máy tính xách tay"],
    "keyboard": ["bàn phím", "keyboard"],
    "mouse": ["chuột máy tính", "mouse"],
    "cell phone": ["điện thoại", "smartphone", "phone", "cell phone"],
    "tv": ["tivi", "tv", "màn hình"],
    "book": ["sách", "book"],
    "bottle": ["chai", "bottle"],
    "cup": ["cốc", "ly", "cup"],
    "sports ball": ["bóng", "quả bóng", "ball"],
    "backpack": ["ba lô", "balo", "backpack"],
    "handbag": ["túi xách", "handbag"],
    "umbrella": ["ô", "dù", "umbrella"],
}


class ObjectClassExtractor:
    def __init__(self, gemini_api_key: str | None = None, text_model_name: str = "gemini-3.5-flash-lite"):
        self.allowed_classes = YOLO_CLASSES
        self.text_model_name = text_model_name
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

    def extract_rule_based(self, text: str) -> list[str]:
        query = text.lower()
        found = []
        for class_name, terms in SYNONYMS.items():
            if class_name in query or any(term.lower() in query for term in terms):
                found.append(class_name)
        return list(dict.fromkeys(found))

    def extract_llm(self, text: str) -> list[str]:
        if not self.client:
            return []
        prompt = (
            "Extract only visible object classes from this video search query. "
            "Return JSON array only. Choose only from this allowed YOLO class list:\n"
            f"{', '.join(self.allowed_classes)}\n\nQuery: {text}"
        )
        response = self.client.models.generate_content(
            model=self.text_model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
            ),
        )
        parsed = json.loads((response.text or "[]").strip())
        if not isinstance(parsed, list):
            return []
        return self._filter_allowed(str(item) for item in parsed)

    def _filter_allowed(self, classes: Iterable[str]) -> list[str]:
        allowed = set(self.allowed_classes)
        result = []
        for item in classes:
            clean = re.sub(r"\s+", " ", item.strip().lower())
            if clean in allowed and clean not in result:
                result.append(clean)
        return result

    def extract(self, text: str, use_llm: bool = True) -> list[str]:
        rule_classes = self.extract_rule_based(text)
        if not use_llm:
            return rule_classes
        try:
            llm_classes = self.extract_llm(text)
        except Exception as exc:
            logger.warning("Không extract object classes bằng Gemini được: %s", exc)
            llm_classes = []
        return self._filter_allowed([*rule_classes, *llm_classes])

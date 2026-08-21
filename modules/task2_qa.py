import json
import logging
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class Task2QAService:
    def __init__(
        self,
        task1_service,
        gemini_api_key: str | None = None,
        videos_dir: str = "videos",
        text_model_name: str = "gemini-3.5-flash-lite",
        vision_model_name: str = "gemini-2.5-flash",
    ):
        self.task1 = task1_service
        self.videos_dir = Path(videos_dir)
        self.text_model_name = os.getenv("GEMINI_TEXT_MODEL", text_model_name)
        self.vision_model_name = os.getenv("GEMINI_VISION_MODEL", vision_model_name)

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY in .env or pass gemini_api_key explicitly."
            )
        self.ai_client = genai.Client(api_key=api_key)

    def _call_gemini_safe(self, contents: list, temperature: float = 0.0) -> str:
        """Gá»i Gemini cÃ³ cÆ¡ cháº¿ thá»­ láº¡i (retry) vÃ  fallback model náº¿u gáº·p Rate Limit (429)"""
        models_to_try = [
            self.vision_model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        # Loáº¡i bá» trÃ¹ng láº·p giá»¯ thá»© tá»±
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for model in models_to_try:
            try:
                response = self.ai_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                if response and response.text:
                    return response.text.strip()
            except errors.APIError as exc:
                if exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                    logger.warning("Model %s bá»‹ quÃ¡ táº£i quota (429). Thá»­ model tiáº¿p theo...", model)
                    time.sleep(1)
                    continue
                logger.warning("Lá»—i API Gemini (%s): %s", model, exc)
            except Exception as e:
                logger.warning("Lá»—i khi gá»i model %s: %s", model, e)
                time.sleep(0.5)

        return "KhÃ´ng thá»ƒ láº¥y cÃ¢u tráº£ lá»i tá»« Gemini API."

    def _parse_query(self, query: str) -> tuple[str, str]:
        """DÃ¹ng Gemini phÃ¢n tÃ­ch query thÃ nh KIS query vÃ  QA question"""
        prompt = f"""
        Báº¡n lÃ  má»™t trá»£ lÃ½ AI xá»­ lÃ½ ngÃ´n ngá»¯ tá»± nhiÃªn.
        Nhiá»‡m vá»¥ cá»§a báº¡n lÃ  phÃ¢n tÃ­ch má»™t cÃ¢u truy váº¥n video thÃ nh 2 pháº§n:
        1. 'kis_query': CÃ¢u mÃ´ táº£ chi tiáº¿t, giÃ u thÃ´ng tin Ä‘á»ƒ dÃ¹ng lÃ m tá»« khÃ³a tÃ¬m kiáº¿m khung hÃ¬nh (frame) trong video.
        2. 'qa_question': CÃ¢u há»i cá»¥ thá»ƒ cáº§n tráº£ lá»i dá»±a trÃªn khung hÃ¬nh Ä‘Ã³.

        Äáº§u vÃ o: "{query}"
        """
        try:
            response = self.ai_client.models.generate_content(
                model=self.text_model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "kis_query": {"type": "STRING"},
                            "qa_question": {"type": "STRING"},
                        },
                        "required": ["kis_query", "qa_question"],
                    },
                ),
            )

            resp_text = (response.text or "").strip()
            parsed = json.loads(resp_text)
            kis_q = parsed.get("kis_query", query).strip()
            qa_q = parsed.get("qa_question", query).strip()
            return (kis_q if kis_q else query, qa_q if qa_q else query)
        except Exception as e:
            logger.warning("Lá»—i khi parse query qua Gemini (%s). DÃ¹ng fallback query gá»‘c.", e)
            return query, query

    def _get_image(self, hit_item: dict) -> Image.Image | None:
        """Láº¥y áº£nh tá»« thÆ° má»¥c cá»¥c bá»™ dá»±a trÃªn payload Qdrant"""
        raw_path = hit_item.get("image_path") or hit_item.get("frame_path")
        if raw_path:
            p = Path(raw_path)
            if p.exists() and p.is_file():
                try:
                    return Image.open(p).convert("RGB")
                except Exception as e:
                    logger.warning("KhÃ´ng thá»ƒ má»Ÿ áº£nh %s: %s", p, e)

            clean_rel = str(raw_path).lstrip("/")
            possible_paths = [
                self.videos_dir / clean_rel,
                Path("data") / clean_rel,
                Path("data/frames") / clean_rel,
                Path("/content/drive/MyDrive/AI_challenge/AIC-1505/data") / clean_rel,
                Path("/content/drive/MyDrive/AI_challenge/AIC-1505") / clean_rel,
                self.videos_dir / clean_rel.replace("data/frames/", "image/"),
                self.videos_dir / clean_rel.replace("data/frames/", ""),
            ]
            for rel_p in possible_paths:
                if rel_p.exists() and rel_p.is_file():
                    try:
                        return Image.open(rel_p).convert("RGB")
                    except Exception as e:
                        logger.warning("KhÃ´ng thá»ƒ má»Ÿ áº£nh tá»« %s: %s", rel_p, e)

        # 2. XÃ¢y dá»±ng Ä‘Æ°á»ng dáº«n Ä‘á»™ng tá»« video_id vÃ  frame_id
        video_id = hit_item.get("video_id")
        frame_id = hit_item.get("frame_id")

        if video_id and frame_id is not None:
            sub_folder = (
                str(video_id).split("_")[0] if "_" in str(video_id) else str(video_id)
            )

            frame_filenames = []
            if isinstance(frame_id, int) or (isinstance(frame_id, str) and str(frame_id).isdigit()):
                f_int = int(frame_id)
                frame_filenames.append(f"{f_int:04d}.jpg")
                frame_filenames.append(f"{f_int:05d}.jpg")
                frame_filenames.append(f"{f_int}.jpg")

            frame_str = str(frame_id)
            if not any(frame_str.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                frame_filenames.append(f"{frame_str}.jpg")
            else:
                frame_filenames.append(frame_str)

            for fname in frame_filenames:
                candidate_paths = [
                    self.videos_dir / "image" / sub_folder / str(video_id) / fname,
                    self.videos_dir / "image" / str(video_id) / fname,
                    Path("data") / "frames" / str(video_id) / fname,
                    Path("data") / sub_folder / str(video_id) / fname,
                    self.videos_dir / sub_folder / str(video_id) / fname,
                    self.videos_dir / str(video_id) / fname,
                    Path("/content/drive/MyDrive/AI_challenge/AIC-1505/data/frames") / str(video_id) / fname,
                ]
                for cp in candidate_paths:
                    if cp.exists() and cp.is_file():
                        try:
                            return Image.open(cp).convert("RGB")
                        except Exception as e:
                            logger.warning("KhÃ´ng thá»ƒ má»Ÿ áº£nh tá»« candidate %s: %s", cp, e)

        # 3. Fallback: Náº¿u cÃ³ frame_url thÃ¬ táº£i trá»±c tiáº¿p
        if hit_item.get("frame_url"):
            try:
                res = requests.get(hit_item["frame_url"], timeout=10)
                if res.status_code == 200:
                    return Image.open(BytesIO(res.content)).convert("RGB")
            except (requests.RequestException, OSError) as exc:
                logger.warning("KhÃ´ng thá»ƒ táº£i áº£nh tá»« URL %s: %s", hit_item.get("frame_url"), exc)

        return None

    def _resolve_existing_path(self, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.exists() and p.is_file():
            return p
        clean_rel = str(raw_path).lstrip("/")
        candidates = [
            self.videos_dir / clean_rel,
            Path("data") / clean_rel,
            Path("data/frames") / clean_rel,
            Path("data/objects") / clean_rel,
            Path("/content/drive/MyDrive/AI_challenge/AIC-1505/data") / clean_rel,
            Path("/content/drive/MyDrive/AI_challenge/AIC-1505") / clean_rel,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _normalize_bbox(self, bbox: Any) -> list[float] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            vals = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
        x1, y1, x2, y2 = vals
        if x2 < x1 or y2 < y1:
            x1, y1, w, h = vals
            x2, y2 = x1 + w, y1 + h
        return [x1, y1, x2, y2]

    def _iter_raw_objects(self, data: Any) -> list[dict]:
        if isinstance(data, list):
            return [obj for obj in data if isinstance(obj, dict)]
        if isinstance(data, dict):
            for key in ["objects", "detections", "boxes", "predictions", "results"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [obj for obj in value if isinstance(obj, dict)]
        return []

    def _get_objects(self, hit_item: dict) -> list[dict]:
        objects_path = hit_item.get("objects_path")
        resolved = self._resolve_existing_path(objects_path)
        if not resolved:
            return []
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Không đọc được YOLO JSON %s: %s", resolved, exc)
            return []

        normalized = []
        for obj in self._iter_raw_objects(data):
            class_name = (
                obj.get("class")
                or obj.get("class_name")
                or obj.get("label")
                or obj.get("name")
            )
            bbox = obj.get("bbox") or obj.get("xyxy") or obj.get("box")
            bbox = self._normalize_bbox(bbox)
            if not class_name or not bbox:
                continue
            confidence = obj.get("confidence", obj.get("conf", obj.get("score")))
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            normalized.append(
                {
                    "class_name": str(class_name).strip().lower(),
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )
        return normalized

    def _select_relevant_objects(
        self, objects: list[dict], question: str, kis_query: str = "", max_objects: int = 8
    ) -> list[dict]:
        if not objects:
            return []
        text = f"{question} {kis_query}".lower()
        matched = [obj for obj in objects if obj.get("class_name", "") in text]
        candidates = matched or objects
        return sorted(
            candidates,
            key=lambda obj: obj.get("confidence") if obj.get("confidence") is not None else 0.0,
            reverse=True,
        )[:max_objects]

    def _crop_objects(
        self, image: Image.Image, objects: list[dict], max_crops: int = 4
    ) -> list[Image.Image]:
        crops = []
        width, height = image.size
        for obj in objects[:max_crops]:
            x1, y1, x2, y2 = obj["bbox"]
            pad = 8
            box = (
                max(0, int(x1) - pad),
                max(0, int(y1) - pad),
                min(width, int(x2) + pad),
                min(height, int(y2) + pad),
            )
            if box[2] > box[0] and box[3] > box[1]:
                crops.append(image.crop(box))
        return crops

    def _annotate_image(self, image: Image.Image, objects: list[dict]) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for idx, obj in enumerate(objects, start=1):
            x1, y1, x2, y2 = obj["bbox"]
            label = f"{idx}:{obj.get('class_name', 'object')}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill="red")
        return annotated

    def _object_summary(self, objects: list[dict]) -> str:
        if not objects:
            return "No YOLO objects available."
        lines = []
        for idx, obj in enumerate(objects, start=1):
            conf = obj.get("confidence")
            conf_text = f", confidence={conf:.3f}" if isinstance(conf, float) else ""
            lines.append(
                f"{idx}. class={obj.get('class_name')}, bbox={obj.get('bbox')}{conf_text}"
            )
        return "\n".join(lines)

    def qa_search(
        self, question: str, top_k: int = 100, vision_top_k: int = 10
    ) -> list[dict]:
        if not question:
            raise ValueError("Can cung cap `question` de thuc hien Task 2.")

        # 1. Phan tich cau hoi thanh kis_query va qa_question
        kis_query, qa_question = self._parse_query(question)
        logger.info("Parsed Query -> KIS: '%s' | QA: '%s'", kis_query, qa_question)

        # 2. Goi Task 1 de dinh vi cac frame lien quan nhat
        candidates = self.task1.find_event(query_description=kis_query, top_k=top_k)
        if not candidates:
            return []

        results = []
        vision_candidates = candidates[: max(1, min(vision_top_k, len(candidates)))]
        logger.info(
            "QA retrieval candidates: %d | Gemini Vision candidates: %d",
            len(candidates),
            len(vision_candidates),
        )
        for cand in vision_candidates:
            img = self._get_image(cand)
            objects = self._get_objects(cand)
            selected_objects = self._select_relevant_objects(objects, qa_question, kis_query)
            object_summary = self._object_summary(selected_objects or objects[:8])

            prompt = (
                "You are answering a visual question about one video frame. "
                "Use the provided image, object crops, and YOLO detections. "
                "Answer extremely briefly and directly. If the question asks for a count, "
                "return only the number. The answer must be <= 100 characters.\n\n"
                f"YOLO detections:\n{object_summary}\n\n"
                f"Question: {qa_question}"
            )

            if img:
                visual_image = self._annotate_image(img, selected_objects) if selected_objects else img
                crops = self._crop_objects(img, selected_objects, max_crops=4)
                contents = [visual_image, *crops, prompt]
            else:
                desc_context = cand.get("desc", "")
                contents = [
                    f"Frame context: {desc_context}\nYOLO detections:\n{object_summary}\n{prompt}"
                ]

            answer_text = self._call_gemini_safe(contents, temperature=0.0)

            results.append(
                {
                    "video_id": cand.get("video_id"),
                    "frame_id": cand.get("frame_id"),
                    "kis_query": kis_query,
                    "qa_question": qa_question,
                    "answer": answer_text,
                    "score": cand.get("score"),
                    "objects_used": selected_objects,
                }
            )

            # Nghỉ nhỏ 0.3s để tránh bị 429 Rate Limit
            time.sleep(0.3)

        return results

    def answer_question(
        self, question: str, top_k: int = 100, vision_top_k: int = 10
    ) -> list[dict]:
        """Alias cho qa_search để tương thích với các script cũ"""
        return self.qa_search(question=question, top_k=top_k, vision_top_k=vision_top_k)

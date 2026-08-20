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
from PIL import Image

logger = logging.getLogger(__name__)


class Task2QAService:
    def __init__(
        self,
        task1_service,
        gemini_api_key: str | None = None,
        videos_dir: str = "videos",
        model_name: str = "gemini-2.5-flash",
    ):
        self.task1 = task1_service
        self.videos_dir = Path(videos_dir)
        self.model_name = os.getenv("GEMINI_MODEL", model_name)

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY in .env or pass gemini_api_key explicitly."
            )
        self.ai_client = genai.Client(api_key=api_key)

    def _call_gemini_safe(self, contents: list, temperature: float = 0.0) -> str:
        """Gọi Gemini có cơ chế thử lại (retry) và fallback model nếu gặp Rate Limit (429)"""
        models_to_try = [
            self.model_name,
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        # Loại bỏ trùng lặp giữ thứ tự
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
                    logger.warning("Model %s bị quá tải quota (429). Thử model tiếp theo...", model)
                    time.sleep(1)
                    continue
                logger.warning("Lỗi API Gemini (%s): %s", model, exc)
            except Exception as e:
                logger.warning("Lỗi khi gọi model %s: %s", model, e)
                time.sleep(0.5)

        return "Không thể lấy câu trả lời từ Gemini API."

    def _parse_query(self, query: str) -> tuple[str, str]:
        """Dùng Gemini phân tích query thành KIS query và QA question"""
        prompt = f"""
        Bạn là một trợ lý AI xử lý ngôn ngữ tự nhiên.
        Nhiệm vụ của bạn là phân tích một câu truy vấn video thành 2 phần:
        1. 'kis_query': Câu mô tả chi tiết, giàu thông tin để dùng làm từ khóa tìm kiếm khung hình (frame) trong video.
        2. 'qa_question': Câu hỏi cụ thể cần trả lời dựa trên khung hình đó.

        Đầu vào: "{query}"
        """
        try:
            response = self.ai_client.models.generate_content(
                model=self.model_name,
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
            logger.warning("Lỗi khi parse query qua Gemini (%s). Dùng fallback query gốc.", e)
            return query, query

    def _get_image(self, hit_item: dict) -> Image.Image | None:
        """Lấy ảnh từ thư mục cục bộ dựa trên payload Qdrant"""
        raw_path = hit_item.get("image_path") or hit_item.get("frame_path")
        if raw_path:
            p = Path(raw_path)
            if p.exists() and p.is_file():
                try:
                    return Image.open(p).convert("RGB")
                except Exception as e:
                    logger.warning("Không thể mở ảnh %s: %s", p, e)

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
                        logger.warning("Không thể mở ảnh từ %s: %s", rel_p, e)

        # 2. Xây dựng đường dẫn động từ video_id và frame_id
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
                            logger.warning("Không thể mở ảnh từ candidate %s: %s", cp, e)

        # 3. Fallback: Nếu có frame_url thì tải trực tiếp
        if hit_item.get("frame_url"):
            try:
                res = requests.get(hit_item["frame_url"], timeout=10)
                if res.status_code == 200:
                    return Image.open(BytesIO(res.content)).convert("RGB")
            except (requests.RequestException, OSError) as exc:
                logger.warning("Không thể tải ảnh từ URL %s: %s", hit_item.get("frame_url"), exc)

        return None

    def qa_search(self, question: str, top_k: int = 1) -> list[dict]:
        if not question:
            raise ValueError("Cần cung cấp ít nhất `question` để thực hiện Task 2.")

        # 1. Phân tích câu hỏi thành kis_query và qa_question
        kis_query, qa_question = self._parse_query(question)
        logger.info("Parsed Query -> KIS: '%s' | QA: '%s'", kis_query, qa_question)

        # 2. Gọi Task 1 để định vị frame liên quan nhất dựa vào kis_query
        candidates = self.task1.find_event(query_description=kis_query, top_k=top_k)
        if not candidates:
            return []

        results = []
        for cand in candidates:
            img = self._get_image(cand)

            # 3. Gọi Gemini để trả lời câu hỏi dựa trên ảnh
            prompt = (
                "Dựa vào hình ảnh được cung cấp từ video, hãy trả lời câu hỏi sau một cách cực kỳ ngắn gọn và trực tiếp. "
                "Đặc biệt nếu là câu hỏi đếm số lượng, CHỈ trả về con số (ví dụ: '5' hoặc 'năm'), tuyệt đối không trả lời thành câu. "
                "Độ dài câu trả lời không vượt quá 100 ký tự.\n"
                f"Câu hỏi: {qa_question}"
            )

            contents = []
            if img:
                contents = [img, prompt]
            else:
                desc_context = cand.get("desc", "")
                contents = [
                    f"Mô tả bối cảnh khung hình video: {desc_context}\n{prompt}"
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
                }
            )

            # Nghỉ nhỏ 0.3s để tránh bị 429 Rate Limit
            time.sleep(0.3)

        # BẮT BUỘC RETURN RESULTS
        return results

    def answer_question(self, question: str, top_k: int = 1) -> list[dict]:
        """Alias cho qa_search để tương thích với các script cũ"""
        return self.qa_search(question=question, top_k=top_k)

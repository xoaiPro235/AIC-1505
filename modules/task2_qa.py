# @title
import logging
import os
import json
from io import BytesIO
from pathlib import Path

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
    ):
        self.task1 = task1_service
        self.videos_dir = Path(videos_dir)

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY in .env or pass gemini_api_key explicitly."
            )
        self.ai_client = genai.Client(api_key=api_key)

    def _parse_query(self, query: str) -> tuple[str, str]:
        """Dùng Gemini phân tích query thành KIS query và QA question"""
        prompt = f"""
        Bạn là một trợ lý AI xử lý ngôn ngữ tự nhiên.
        Nhiệm vụ của bạn là phân tích một câu truy vấn video thành 2 phần:
        1. 'kis_query': Câu mô tả chi tiết, giàu thông tin để dùng làm từ khóa tìm kiếm khung hình (frame) trong video. Hãy sắp xếp lại hoặc bổ sung thêm ngữ cảnh nếu cần.
        2. 'qa_question': Câu hỏi cụ thể cần trả lời dựa trên khung hình đó.

        Đầu vào: "{query}"
        """
        try:
            response = self.ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "kis_query": {"type": "STRING"},
                            "qa_question": {"type": "STRING"}
                        },
                        "required": ["kis_query", "qa_question"]
                    }
                ),
            )

            parsed = json.loads((response.text or "").strip())
            return parsed.get('kis_query', query), parsed.get('qa_question', query)
        except Exception as e:
            logger.warning(f"Lỗi khi parse query: {e}")
            return query, query

    def _get_image(self, hit_item: dict) -> Image.Image | None:
        """
        Lấy ảnh từ thư mục cục bộ dựa trên payload Qdrant
        """
        raw_path = hit_item.get("image_path") or hit_item.get("frame_path")
        if raw_path:
            if raw_path.startswith("/"):
                raw_path = raw_path[1:]

            possible_paths = [
                self.videos_dir / raw_path,
                Path("/content/drive/MyDrive/AI_challenge/AIC-1505/data") / raw_path,
                Path("/content/drive/MyDrive/AI_challenge/AIC-1505") / raw_path
            ]

            for p in possible_paths:
                if p.exists():
                    try:
                        return Image.open(p).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Không thể mở ảnh {p}: {e}")
        return None

    def qa_search(self, question: str, top_k: int = 1) -> list[dict]:
        if not question:
            raise ValueError("Cần cung cấp ít nhất `question` để thực hiện Task 2.")

        # 1. Phân tích câu hỏi thành kis_query và qa_question
        kis_query, qa_question = self._parse_query(question)
        logger.info(f"Parsed Query -> KIS: {kis_query} | QA: {qa_question}")

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
                "Đặc biệt nếu là câu hỏi đếm số lượng, CHỈ trả về con số (ví dụ: '5' hoặc 'năm'), tuyệt đối không trả lời thành câu.\n"
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

            try:
                response = self.ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                answer_text = response.text.strip() if response and response.text else "Không có câu trả lời."
            except (errors.APIError, Exception) as exc:
                logger.warning("Gemini API call failed for frame %s: %s", cand.get("frame_id"), exc)
                answer_text = "Không thể lấy câu trả lời từ Gemini API."

            results.append(
                {
                    "video_id": cand.get("video_id"),
                    "frame_id": cand.get("frame_id"),
                    "kis_query": kis_query,
                    "qa_question": qa_question,
                    "answer": answer_text,
                }
            )

    def answer_question(self, question: str, top_k: int = 1) -> list[dict]:
        """Alias cho qa_search để tương thích với các script cũ"""
        return self.qa_search(question=question, top_k=top_k)

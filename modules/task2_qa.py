import logging
import os
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

    def _extract_visual_description(self, question: str) -> str:
        """Dùng Gemini tách/biến câu hỏi thành mô tả thị giác (Visual Description) để query vector DB"""
        prompt = (
            "Dựa vào câu hỏi dưới đây, hãy tạo ra một câu mô tả ngắn gọn về khung cảnh/hình ảnh thị giác (visual description) "
            "cần tìm trong video để có thể trả lời câu hỏi này.\n"
            f"Câu hỏi: {question}\n"
            "Chỉ trả về 1 câu mô tả ngắn gọn, không giải thích gì thêm."
        )
        response = self.ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text.strip()

    def _get_image(self, hit_item: dict) -> Image.Image | None:
        """
        Lấy ảnh từ thư mục cục bộ dựa trên payload Qdrant:
        - image_path: "/data/frames/L01_V001/0105.jpg"
        - Hoặc ghép theo: videos/image/<sub_folder>/<video_id>/<frame_id>.jpg
        """
        # 1. Kiểm tra image_path / frame_path từ payload Qdrant
        raw_path = hit_item.get("image_path") or hit_item.get("frame_path")
        if raw_path:
            p = Path(raw_path)
            if p.exists():
                return Image.open(p)
            # Thử ghép đường dẫn relative nếu raw_path có dạng /data/frames/...
            clean_rel = str(raw_path).lstrip("/")
            possible_rel_paths = [
                Path(clean_rel),
                self.videos_dir / clean_rel,
                self.videos_dir / clean_rel.replace("data/frames/", "image/"),
                self.videos_dir / clean_rel.replace("data/frames/", ""),
            ]
            for rel_p in possible_rel_paths:
                if rel_p.exists():
                    return Image.open(rel_p)

        # 2. Xây dựng đường dẫn động từ video_id và frame_id
        video_id = hit_item.get("video_id")
        frame_id = hit_item.get("frame_id")

        if video_id and frame_id is not None:
            sub_folder = str(video_id).split("_")[0] if "_" in str(video_id) else str(video_id)
            
            # Thử các định dạng frame_id (ví dụ: 105 -> "0105.jpg" và "105.jpg")
            frame_filenames = []
            if isinstance(frame_id, int) or (isinstance(frame_id, str) and frame_id.isdigit()):
                frame_filenames.append(f"{int(frame_id):04d}.jpg")
                frame_filenames.append(f"{int(frame_id)}.jpg")
            
            frame_str = str(frame_id)
            if not any(frame_str.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]):
                frame_filenames.append(f"{frame_str}.jpg")
            else:
                frame_filenames.append(frame_str)

            for fname in frame_filenames:
                possible_paths = [
                    self.videos_dir / "image" / sub_folder / str(video_id) / fname,
                    self.videos_dir / "image" / str(video_id) / fname,
                    self.videos_dir / "data" / "frames" / str(video_id) / fname,
                    self.videos_dir / sub_folder / str(video_id) / fname,
                ]
                for p in possible_paths:
                    if p.exists():
                        return Image.open(p)

        # 3. Fallback: Nếu có frame_url
        if hit_item.get("frame_url"):
            try:
                res = requests.get(hit_item["frame_url"], timeout=10)
                if res.status_code == 200:
                    return Image.open(BytesIO(res.content))
            except (requests.RequestException, OSError) as exc:
                logger.warning("Failed to fetch or open image from URL: %s", exc)

        return None

    def answer_question(
        self, description: str | None = None, question: str | None = None, top_k: int = 1
    ) -> list[dict]:
        if not question and description:
            # Trường hợp người dùng truyền 1 tham số duy nhất là câu hỏi vào vị trí description
            question = description
            description = None

        if not question:
            raise ValueError("Cần cung cấp ít nhất `question` để thực hiện Task 2.")

        # Tự động biến câu hỏi thành câu miêu tả nếu chưa có description
        search_description = description or self._extract_visual_description(question)

        # 1. Gọi Task 1 để định vị frame liên quan nhất dựa vào search_description
        candidates = self.task1.find_event(query_description=search_description, top_k=top_k)
        if not candidates:
            return []

        results = []
        for cand in candidates:
            img = self._get_image(cand)

            # 2. Gọi Gemini để trả lời câu hỏi dựa trên ảnh (hoặc bối cảnh payload nếu chưa có ảnh)
            prompt = (
                "Dựa vào hình ảnh được cung cấp từ video, hãy trả lời câu hỏi sau thật ngắn gọn:\n"
                f"Câu hỏi: {question}"
            )

            contents = []
            if img:
                contents = [prompt, img]
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
            except (errors.APIError, Exception) as exc:  # noqa: BLE001
                logger.warning("Gemini API call failed for frame %s: %s", cand.get("frame_id"), exc)
                answer_text = "Không thể lấy câu trả lời từ Gemini API."

            results.append(
                {
                    "video_id": cand["video_id"],
                    "frame_id": cand["frame_id"],
                    "search_description": search_description,
                    "answer": answer_text,
                }
            )

        return results


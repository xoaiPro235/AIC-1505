import json

from config import get_env
from core.db_client import QdrantService
from core.text_encoder import SigLIPEncoder
from modules.task1_kis import Task1KISService
from modules.task2_qa import Task2QAService
from modules.task3_trake import Task3TRAKEService

# 1. Khởi tạo kết nối DB Qdrant
DB_URL = get_env("QDRANT_URL")
DB_HOST = get_env("QDRANT_HOST", "localhost")
DB_PORT = int(get_env("QDRANT_PORT", "6333"))
DB_API_KEY = get_env("QDRANT_API_KEY")
DB_COLLECTION = get_env("QDRANT_COLLECTION", "aic2026_clip_v1")
HF_TOKEN = get_env("HF_TOKEN")

db_service = QdrantService(
    url=DB_URL,
    host=DB_HOST,
    port=DB_PORT,
    api_key=DB_API_KEY,
    collection_name=DB_COLLECTION,
)

# 2. Khởi tạo bộ Text Encoder (SigLIP)
text_encoder = SigLIPEncoder()

# 3. Tạo instance cho Task 1 (Textual KIS)
task1 = Task1KISService(db_service=db_service, text_encoder=text_encoder)

# 4. Tạo instance cho Task 2 (Q&A)
gemini_key = get_env("GEMINI_API_KEY")
if gemini_key and gemini_key != "your_gemini_api_key_here":
    task2 = Task2QAService(
        task1_service=task1,
        gemini_api_key=gemini_key,
        videos_dir="videos",
    )
else:
    task2 = None

# 5. Tạo instance cho Task 3 (TRAKE)
task3 = Task3TRAKEService(task1_service=task1)

if __name__ == "__main__":
    # # --- Kiểm thử Task 1: Textual KIS ---
    # print("\n==========================================")
    # print("=== Dạng 1: Textual KIS ===")
    # print("==========================================")
    # res_task1 = task1.find_event("Một người đang mở laptop trong văn phòng", top_k=10)
    # print(json.dumps(res_task1, indent=2, ensure_ascii=False))

    # --- Kiểm thử Task 2: Q&A ---
    print("\n==========================================")
    print("=== Dạng 2: Hỏi - Đáp (Q&A) ===")
    print("==========================================")
    if task2:
        res_task2 = task2.qa_search(
            question="Một người đang dùng laptop trong văn phòng, laptop đó có màu gì?",
            top_k=3,
        )
        print(json.dumps(res_task2, indent=2, ensure_ascii=False))
    else:
        print("[LƯU Ý] Chưa điền GEMINI_API_KEY trong .env. Hãy điền key để chạy Task 2 VLM.")

    # --- Kiểm thử Task 3: TRAKE ---
    print("\n==========================================")
    print("=== Dạng 3: TRAKE (Temporal Retrieval & Alignment) ===")
    print("==========================================")
    events_query = [
        "Vận động viên bắt đầu chạy đà",
        "Vận động viên giậm nhảy rời khỏi mặt đất",
        "Vận động viên bay qua xà ngang",
        "Vận động viên tiếp đất lên đệm",
    ]
    res_task3 = task3.align_events(events_query, top_k_results=3)
    print(json.dumps(res_task3, indent=2, ensure_ascii=False))

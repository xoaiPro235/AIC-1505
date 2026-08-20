import json
from pathlib import Path

from config import get_env
from core.db_client import QdrantService
from core.text_encoder import SigLIPEncoder
from modules.task1_kis import Task1KISService
from modules.task2_qa import Task2QAService
from modules.task3_trake import Task3TRAKEService
from utils.formatter import (
    create_submission_zip,
    export_kis_csv,
    export_qa_csv,
    export_trake_csv,
    validate_submission,
)

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
    # Thư mục chứa các file submission nộp BTC
    sub_dir = Path("submission")
    sub_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("      AIC 2026 - CHẠY THỬ NGHIỆM 3 DẠNG BÀI THI")
    print("=" * 50)

    # --- Dạng 1: Textual KIS ---
    print("\n[1] Đang chạy Task 1: Textual KIS...")
    query_kis = "Một người đang mở laptop trong văn phòng"
    res_task1 = task1.find_event(query_kis, top_k=100)
    # Tự động xuất CSV (tự cộng +1 offset frame theo chuẩn BTC)
    export_kis_csv(res_task1, sub_dir / "query-1-kis.csv", max_rows=100)
    print(f"-> Đã tìm thấy {len(res_task1)} kết quả và xuất vào {sub_dir}/query-1-kis.csv")

    # --- Dạng 2: Hỏi - Đáp (Q&A) ---
    print("\n[2] Đang chạy Task 2: Hỏi - Đáp (Q&A)...")
    if task2:
        query_qa = "Một người đang dùng laptop trong văn phòng, laptop đó có màu gì?"
        res_task2 = task2.qa_search(question=query_qa, top_k=3)
        export_qa_csv(res_task2, sub_dir / "query-2-qa.csv", max_rows=100)
        print(f"-> Đã trả lời {len(res_task2) if res_task2 else 0} kết quả và xuất vào {sub_dir}/query-2-qa.csv")
    else:
        print("-> [BỎ QUA] Chưa cấu hình GEMINI_API_KEY trong .env")

    # --- Dạng 3: TRAKE (Temporal Retrieval & Alignment) ---
    print("\n[3] Đang chạy Task 3: TRAKE...")
    events_query = [
        "Vận động viên bắt đầu chạy đà",
        "Vận động viên giậm nhảy rời khỏi mặt đất",
        "Vận động viên bay qua xà ngang",
        "Vận động viên tiếp đất lên đệm",
    ]
    res_task3 = task3.align_events(events_query, top_k_results=10)
    export_trake_csv(
        res_task3,
        sub_dir / "query-3-trake.csv",
        expected_events_count=len(events_query),
        max_rows=100,
    )
    print(f"-> Đã căn chỉnh {len(res_task3)} video và xuất vào {sub_dir}/query-3-trake.csv")

    # --- Kiểm tra Checklist & Đóng gói ZIP ---
    print("\n" + "=" * 50)
    print("      KIỂM TRA & ĐÓNG GÓI SUBMISSION (.ZIP)")
    print("=" * 50)
    validate_submission(sub_dir)
    zip_file = create_submission_zip(sub_dir, "submission.zip")
    print(f"-> ĐÃ TẠO FILE NỘP BÀI THÀNH CÔNG: {zip_file.resolve()}\n")

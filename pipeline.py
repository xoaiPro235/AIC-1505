import argparse
import logging
import sys
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
    parse_query_file,
    validate_submission,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AICPipeline")


class AICPipeline:
    """Pipeline tự động hóa xử lý toàn bộ gói truy vấn của Ban tổ chức AIC:
    1. Đọc file query (.txt) và nhận diện dạng bài (KIS, QA, TRAKE)
    2. Chạy module tương ứng để tìm kiếm và sinh đáp án
    3. Định dạng và xuất file .csv chuẩn vào thư mục submission/
    4. Kiểm tra Checklist định dạng (Validator)
    5. Đóng gói thư mục submission/ thành file .zip để nộp bài
    """

    def __init__(self):
        # 1. Khởi tạo Qdrant
        db_url = get_env("QDRANT_URL")
        db_host = get_env("QDRANT_HOST", "localhost")
        db_port = int(get_env("QDRANT_PORT", "6333"))
        db_api_key = get_env("QDRANT_API_KEY")
        db_collection = get_env("QDRANT_COLLECTION", "aic2026_clip_v1")

        logger.info("Đang kết nối Qdrant (%s:%s)...", db_url or db_host, db_port)
        self.db_service = QdrantService(
            url=db_url,
            host=db_host,
            port=db_port,
            api_key=db_api_key,
            collection_name=db_collection,
        )

        # 2. Khởi tạo Text Encoder
        logger.info("Đang nạp bộ Text Encoder (SigLIP)...")
        self.text_encoder = SigLIPEncoder()

        # 3. Khởi tạo Task 1 (Textual KIS)
        self.task1 = Task1KISService(
            db_service=self.db_service, text_encoder=self.text_encoder
        )

        # 4. Khởi tạo Task 2 (Q&A)
        gemini_key = get_env("GEMINI_API_KEY")
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            self.task2 = Task2QAService(
                task1_service=self.task1,
                gemini_api_key=gemini_key,
                videos_dir="videos",
            )
        else:
            logger.warning(
                "GEMINI_API_KEY chưa được thiết lập. Task 2 Q&A sẽ không sử dụng được Gemini VLM."
            )
            self.task2 = None

        # 5. Khởi tạo Task 3 (TRAKE)
        self.task3 = Task3TRAKEService(task1_service=self.task1)

    def process_single_query(
        self,
        query_file_path: str | Path,
        output_dir: str | Path = "submission",
        top_k: int = 100,
    ) -> Path:
        """Xử lý 1 file query .txt và xuất ra file .csv tương ứng."""
        query_info = parse_query_file(query_file_path)
        q_type = query_info["query_type"]
        csv_name = query_info["output_csv_name"]
        output_csv_path = Path(output_dir) / csv_name

        logger.info(
            "--- Đang xử lý: %s [Loại: %s] ---",
            query_info["query_id"],
            q_type.upper(),
        )

        if q_type == "kis":
            desc = query_info.get("kis_description", "")
            logger.info("Query KIS: '%s'", desc)
            results = self.task1.find_event(query_description=desc, top_k=top_k)
            export_kis_csv(results, output_csv_path, max_rows=top_k)

        elif q_type == "qa":
            question = query_info.get("qa_question", "")
            logger.info("Query Q&A: '%s'", question)
            if not self.task2:
                raise RuntimeError(
                    "Cần có GEMINI_API_KEY trong .env để chạy Task 2 (Q&A)."
                )
            results = self.task2.qa_search(
                question=question, top_k=min(top_k, 20)
            )
            export_qa_csv(results, output_csv_path, max_rows=top_k)

        elif q_type == "trake":
            events = query_info.get("trake_events", [])
            logger.info("Query TRAKE với %d events: %s", len(events), events)
            results = self.task3.align_events(
                events, top_k_per_event=150, top_k_results=top_k
            )
            export_trake_csv(
                results,
                output_csv_path,
                expected_events_count=len(events),
                max_rows=top_k,
            )

        return output_csv_path

    def run_batch(
        self,
        queries_dir: str | Path = "queries",
        output_dir: str | Path = "submission",
        output_zip: str | Path = "submission.zip",
        top_k: int = 100,
    ) -> Path:
        """Xử lý toàn bộ các file query .txt trong thư mục queries_dir,
        sau đó kiểm tra hợp lệ và nén thành file submission .zip.
        """
        q_dir = Path(queries_dir)
        if not q_dir.exists():
            raise FileNotFoundError(f"Thư mục queries không tồn tại: {q_dir.resolve()}")

        query_files = sorted(list(q_dir.glob("*.txt")))
        if not query_files:
            raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {q_dir}")

        logger.info(
            "Tìm thấy %d file query trong thư mục '%s'", len(query_files), q_dir
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for q_file in query_files:
            try:
                self.process_single_query(q_file, output_dir=out_dir, top_k=top_k)
            except Exception as e:
                logger.error("Lỗi khi xử lý file %s: %s", q_file.name, e)

        # 1. Kiểm tra Checklist
        validation_report = validate_submission(out_dir)

        # 2. Đóng gói ZIP
        zip_path = create_submission_zip(out_dir, output_zip)

        if validation_report["is_valid"]:
            logger.info(
                "ĐÓNG GÓI HOÀN TẤT! Sẵn sàng nộp bài: %s", zip_path.resolve()
            )
        else:
            logger.warning(
                "Đã tạo file zip nhưng có cảnh báo/lỗi định dạng! Hãy kiểm tra lại output trên."
            )

        return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="AIC 2026 Batch Query & Submission Runner"
    )
    parser.add_argument(
        "--queries_dir",
        "-q",
        type=str,
        default="queries",
        help="Thư mục chứa các file query (.txt) từ BTC",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="submission",
        help="Thư mục xuất các file CSV submission",
    )
    parser.add_argument(
        "--zip_name",
        "-z",
        type=str,
        default="submission.zip",
        help="Tên file zip nộp bài",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=100,
        help="Số dòng dự đoán tối đa cho mỗi query (tối đa 100)",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Chỉ kiểm tra định dạng thư mục submission mà không chạy model",
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_submission(args.output_dir)
        create_submission_zip(args.output_dir, args.zip_name)
        return

    pipeline = AICPipeline()
    pipeline.run_batch(
        queries_dir=args.queries_dir,
        output_dir=args.output_dir,
        output_zip=args.zip_name,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

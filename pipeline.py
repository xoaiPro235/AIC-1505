import argparse
import logging
import sys
from pathlib import Path

from config import get_env
from core.db_client import QdrantService
from core.text_encoder import SigLIPEncoder
from modules.object_extractor import ObjectClassExtractor
from modules.query_rewriter import QueryRewriter
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
    """Batch runner for AIC KIS, QA, and TRAKE submissions."""

    def __init__(
        self,
        extract_objects: bool = False,
        filter_mode: str = "should",
        trake_candidate_videos: int = 10,
        qa_vision_top_k: int = 10,
        rewrite_query: bool = False,
    ):
        db_url = get_env("QDRANT_URL")
        db_host = get_env("QDRANT_HOST", "localhost")
        db_port = int(get_env("QDRANT_PORT", "6333"))
        db_api_key = get_env("QDRANT_API_KEY")
        db_collection = get_env("QDRANT_COLLECTION", "aic2026_siglip2_so400m_v1")

        logger.info("Connecting Qdrant (%s:%s)...", db_url or db_host, db_port)
        self.db_service = QdrantService(
            url=db_url,
            host=db_host,
            port=db_port,
            api_key=db_api_key,
            collection_name=db_collection,
        )

        logger.info("Loading Text Encoder (SigLIP2 SO400M)...")
        self.text_encoder = SigLIPEncoder()
        self.extract_objects = extract_objects
        self.filter_mode = filter_mode
        self.trake_candidate_videos = trake_candidate_videos
        self.qa_vision_top_k = qa_vision_top_k
        self.rewrite_query = rewrite_query

        gemini_key = get_env("GEMINI_API_KEY")
        gemini_text_model = get_env("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite")
        gemini_vision_model = get_env("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        self.object_extractor = ObjectClassExtractor(
            gemini_api_key=gemini_key,
            text_model_name=gemini_text_model or "gemini-3.5-flash-lite",
        )
        self.query_rewriter = QueryRewriter(
            gemini_api_key=gemini_key,
            text_model_name=gemini_text_model or "gemini-3.5-flash-lite",
        )

        self.task1 = Task1KISService(
            db_service=self.db_service,
            text_encoder=self.text_encoder,
            object_extractor=self.object_extractor,
            enable_extract=extract_objects,
            filter_mode=filter_mode,
            query_rewriter=self.query_rewriter,
            rewrite_query=rewrite_query,
        )

        if gemini_key and gemini_key != "your_gemini_api_key_here":
            self.task2 = Task2QAService(
                task1_service=self.task1,
                gemini_api_key=gemini_key,
                videos_dir="videos",
                text_model_name=gemini_text_model or "gemini-3.5-flash-lite",
                vision_model_name=gemini_vision_model or "gemini-2.5-flash",
            )
        else:
            logger.warning("GEMINI_API_KEY is not set. Task 2 Q&A will be unavailable.")
            self.task2 = None

        self.task3 = Task3TRAKEService(task1_service=self.task1)

    def process_single_query(
        self,
        query_file_path: str | Path,
        output_dir: str | Path = "submission",
        top_k: int = 100,
    ) -> Path:
        query_info = parse_query_file(query_file_path)
        q_type = query_info["query_type"]
        output_csv_path = Path(output_dir) / query_info["output_csv_name"]

        logger.info("--- Processing: %s [Type: %s] ---", query_info["query_id"], q_type.upper())

        if q_type == "kis":
            desc = query_info.get("kis_description", "")
            results = self.task1.find_event(
                query_description=desc,
                top_k=top_k,
                rewrite=self.rewrite_query,
            )
            export_kis_csv(results, output_csv_path, max_rows=top_k)

        elif q_type == "qa":
            question = query_info.get("qa_question", "")
            if not self.task2:
                raise RuntimeError("GEMINI_API_KEY is required for Task 2 Q&A.")
            results = self.task2.qa_search(
                question=question,
                top_k=top_k,
                vision_top_k=self.qa_vision_top_k,
            )
            export_qa_csv(results, output_csv_path, max_rows=top_k)

        elif q_type == "trake":
            events = query_info.get("trake_events", [])
            results = self.task3.align_events(
                events,
                top_k_per_event=150,
                top_k_results=top_k,
                candidate_videos=self.trake_candidate_videos,
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
        q_dir = Path(queries_dir)
        if not q_dir.exists():
            raise FileNotFoundError(f"Queries directory does not exist: {q_dir.resolve()}")

        query_files = sorted(q_dir.glob("*.txt"))
        if not query_files:
            raise FileNotFoundError(f"No .txt query files found in: {q_dir}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for q_file in query_files:
            try:
                self.process_single_query(q_file, output_dir=out_dir, top_k=top_k)
            except Exception as exc:
                logger.error("Error processing %s: %s", q_file.name, exc)

        validation_report = validate_submission(out_dir)
        zip_path = create_submission_zip(out_dir, output_zip)
        if validation_report["is_valid"]:
            logger.info("Submission package ready: %s", zip_path.resolve())
        else:
            logger.warning("Zip created, but validator reported errors/warnings.")
        return zip_path


def main():
    parser = argparse.ArgumentParser(description="AIC 2026 Batch Query & Submission Runner")
    parser.add_argument("--queries_dir", "-q", default="queries", help="Directory containing query .txt files")
    parser.add_argument("--output_dir", "-o", default="submission", help="Directory for CSV outputs")
    parser.add_argument("--zip_name", "-z", default="submission.zip", help="Output zip file name")
    parser.add_argument("--top_k", "-k", type=int, default=100, help="Max predictions per query")
    parser.add_argument("--validate_only", action="store_true", help="Validate submission dir only")
    parser.add_argument("--extract", action="store_true", help="Extract object classes for Qdrant pre-filtering")
    parser.add_argument("--filter_mode", choices=["should", "must"], default="should", help="Object filter logic")
    parser.add_argument("--trake_candidate_videos", type=int, default=10, help="Candidate videos for TRAKE DP")
    parser.add_argument("--qa_vision_top_k", type=int, default=10, help="QA candidates sent to Gemini Vision")
    parser.add_argument("--rewrite", action="store_true", help="Auto rewrite long/complex queries before SigLIP2 encoding")
    args = parser.parse_args()

    if args.validate_only:
        validate_submission(args.output_dir)
        create_submission_zip(args.output_dir, args.zip_name)
        return

    pipeline = AICPipeline(
        extract_objects=args.extract,
        filter_mode=args.filter_mode,
        trake_candidate_videos=args.trake_candidate_videos,
        qa_vision_top_k=args.qa_vision_top_k,
        rewrite_query=args.rewrite,
    )
    pipeline.run_batch(
        queries_dir=args.queries_dir,
        output_dir=args.output_dir,
        output_zip=args.zip_name,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

import logging

logger = logging.getLogger(__name__)


class Task1KISService:
    def __init__(
        self,
        db_service,
        text_encoder,
        object_extractor=None,
        enable_extract: bool = False,
        filter_mode: str = "should",
    ):
        # Tiêm (Inject) kết nối DB, Encoder và optional Object Extractor vào
        self.db = db_service
        self.encoder = text_encoder
        self.object_extractor = object_extractor
        self.enable_extract = enable_extract
        self.filter_mode = filter_mode

    def find_event(
        self,
        query_description: str,
        object_filter: list[str] | None = None,
        top_k: int = 5,
        extract: bool | None = None,
    ) -> list[dict]:
        # Bước 1: Biến mô tả thành vector SigLIP2
        query_vector = self.encoder.encode(query_description)

        # Bước 2: Nếu bật --extract thì trích object classes để pre-filter Qdrant
        should_extract = self.enable_extract if extract is None else extract
        resolved_filter = object_filter
        if resolved_filter is None and should_extract and self.object_extractor:
            resolved_filter = self.object_extractor.extract(query_description, use_llm=True)
            if resolved_filter:
                logger.info("Object filter extracted: %s", resolved_filter)

        # Bước 3: Nhờ Qdrant tìm top frames, có fallback dense-only nếu filter rỗng kết quả
        return self.db.query_by_vector(
            vector=query_vector,
            object_filter=resolved_filter,
            top_k=top_k,
            filter_mode=self.filter_mode,
            fallback_without_filter=True,
        )

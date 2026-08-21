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
        query_rewriter=None,
        rewrite_query: bool = False,
    ):
        # Tiêm (Inject) kết nối DB, Encoder và optional Object Extractor vào
        self.db = db_service
        self.encoder = text_encoder
        self.object_extractor = object_extractor
        self.enable_extract = enable_extract
        self.filter_mode = filter_mode
        self.query_rewriter = query_rewriter
        self.rewrite_query = rewrite_query

    def find_event(
        self,
        query_description: str,
        object_filter: list[str] | None = None,
        top_k: int = 5,
        extract: bool | None = None,
        rewrite: bool | None = None,
    ) -> list[dict]:
        # Bước 1: Nếu bật --rewrite và query dài/phức tạp, compact trước khi encode SigLIP2
        should_rewrite = self.rewrite_query if rewrite is None else rewrite
        search_query = query_description
        if should_rewrite and self.query_rewriter:
            search_query = self.query_rewriter.rewrite(query_description)

        # Bước 2: Biến mô tả tìm kiếm thành vector SigLIP2
        query_vector = self.encoder.encode(search_query)

        # Bước 2: Nếu bật --extract thì trích object classes để pre-filter Qdrant
        should_extract = self.enable_extract if extract is None else extract
        resolved_filter = object_filter
        if resolved_filter is None and should_extract and self.object_extractor:
            resolved_filter = self.object_extractor.extract(query_description, use_llm=True)
            if resolved_filter:
                logger.info("Object filter extracted: %s", resolved_filter)

        results = self.db.query_by_vector(
            vector=query_vector,
            object_filter=resolved_filter,
            top_k=top_k,
            filter_mode=self.filter_mode,
            fallback_without_filter=True,
        )
        for item in results:
            item["search_query"] = search_query
            item["original_query"] = query_description
        return results

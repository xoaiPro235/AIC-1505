class Task1KISService:
    def __init__(self, db_service, text_encoder):
        # Tiêm (Inject) kết nối DB và Encoder vào
        self.db = db_service
        self.encoder = text_encoder

    def find_event(
        self, query_description: str, object_filter: list[str] | None = None, top_k: int = 5
    ) -> list[dict]:
        # Bước 1: Biến mô tả thành vector
        query_vector = self.encoder.encode(query_description)
        # Bước 2: Nhờ Qdrant tìm top frames
        return self.db.query_by_vector(
            vector=query_vector, object_filter=object_filter, top_k=top_k
        )

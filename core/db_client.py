from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantService:
    def __init__(
        self,
        host: str | None = None,
        port: int = 6333,
        api_key: str | None = None,
        collection_name: str = "aic2026_clip_v1",
        https: bool = False,
        url: str | None = None,
    ):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key, https=https)
        self.collection_name = collection_name

    def query_by_vector(
        self, vector: list[float], object_filter: list[str] | None = None, top_k: int = 5
    ) -> list[dict]:
        """Chỉ làm nhiệm vụ nhận vector và trả về kết quả từ DB"""
        query_filter = None
        if object_filter:
            conditions = [
                models.FieldCondition(
                    key="object_classes", match=models.MatchValue(value=obj)
                )
                for obj in object_filter
            ]
            query_filter = models.Filter(must=conditions)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "video_id": payload.get("video_id"),
                    "frame_id": payload.get("frame_id"),
                    "score": hit.score,
                    "image_path": payload.get("image_path"),
                    "objects_path": payload.get("objects_path"),
                    "object_classes": payload.get("object_classes", []),
                    "desc": payload.get("desc"),
                    "frame_path": payload.get("image_path") or payload.get("frame_path"),
                    "frame_url": payload.get("frame_url"),
                }
            )
        return results


from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantService:
    def __init__(
        self,
        host: str | None = None,
        port: int = 6333,
        api_key: str | None = None,
        collection_name: str = "aic2026_siglip2_so400m_v1",
        https: bool = False,
        url: str | None = None,
    ):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key, https=https)
        self.collection_name = collection_name

    def _build_object_filter(
        self, object_filter: list[str] | None, filter_mode: str = "should"
    ) -> models.Filter | None:
        clean_objects = [str(obj).strip() for obj in (object_filter or []) if str(obj).strip()]
        if not clean_objects:
            return None

        conditions = [
            models.FieldCondition(
                key="object_classes", match=models.MatchValue(value=obj)
            )
            for obj in dict.fromkeys(clean_objects)
        ]

        if filter_mode == "must":
            return models.Filter(must=conditions)  # type: ignore[arg-type]
        return models.Filter(should=conditions)  # type: ignore[arg-type]

    def _extract_vector(self, point: Any) -> list[float] | None:
        vector = getattr(point, "vector", None)
        if vector is None and isinstance(point, dict):
            vector = point.get("vector")
        if isinstance(vector, dict):
            first_vector = next(iter(vector.values()), None)
            return first_vector if isinstance(first_vector, list) else None
        return vector if isinstance(vector, list) else None

    def _point_to_result(self, point: Any) -> dict:
        payload = getattr(point, "payload", None) or {}
        score = getattr(point, "score", None)
        vector = self._extract_vector(point)
        return {
            "video_id": payload.get("video_id"),
            "frame_id": payload.get("frame_id"),
            "score": score,
            "image_path": payload.get("image_path"),
            "objects_path": payload.get("objects_path"),
            "object_classes": payload.get("object_classes", []),
            "desc": payload.get("desc"),
            "frame_path": payload.get("image_path") or payload.get("frame_path"),
            "frame_url": payload.get("frame_url"),
            "vector": vector,
            "payload": payload,
        }

    def _query_points(
        self,
        vector: list[float],
        query_filter: models.Filter | None,
        top_k: int,
    ) -> list[Any]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return list(response.points)

        return list(
            self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        )  # type: ignore[attr-defined]

    def query_by_vector(
        self,
        vector: list[float],
        object_filter: list[str] | None = None,
        top_k: int = 5,
        filter_mode: str = "should",
        fallback_without_filter: bool = True,
    ) -> list[dict]:
        """Nhận vector và trả về kết quả từ Qdrant.

        `filter_mode="should"` nghĩa là frame chứa ít nhất một object class.
        `filter_mode="must"` nghĩa là frame phải chứa tất cả object classes.
        """
        query_filter = self._build_object_filter(object_filter, filter_mode)
        hits = self._query_points(vector=vector, query_filter=query_filter, top_k=top_k)

        if not hits and query_filter is not None and fallback_without_filter:
            hits = self._query_points(vector=vector, query_filter=None, top_k=top_k)

        return [self._point_to_result(hit) for hit in hits]

    def get_frames_by_video(
        self,
        video_id: str,
        limit: int | None = None,
        batch_size: int = 256,
        with_vectors: bool = True,
    ) -> list[dict]:
        """Tải toàn bộ frame thuộc một video từ Qdrant để phục vụ TRAKE alignment."""
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ]
        )

        points: list[Any] = []
        offset = None
        while True:
            current_limit = batch_size
            if limit is not None:
                remaining = limit - len(points)
                if remaining <= 0:
                    break
                current_limit = min(current_limit, remaining)

            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=current_limit,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            points.extend(batch)
            if offset is None:
                break

        results = [self._point_to_result(point) for point in points]

        def frame_sort_key(item: dict) -> int:
            try:
                return int(float(str(item.get("frame_id", 0))))
            except (TypeError, ValueError):
                return 0

        return sorted(results, key=frame_sort_key)

import logging
from collections import defaultdict
from math import inf

logger = logging.getLogger(__name__)


class Task3TRAKEService:
    def __init__(self, task1_service):
        self.task1 = task1_service
        self.db = task1_service.db
        self.encoder = task1_service.encoder

    def _dot(self, a: list[float], b: list[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    def _fallback_from_hits(self, video_id: str, events: list[str], video_stats: dict) -> dict:
        best_frames = []
        current_min_frame = -1
        for event_idx in range(len(events)):
            frames = video_stats[video_id]["events"].get(event_idx, [])
            frames = sorted(frames, key=lambda x: x[1], reverse=True)
            selected = current_min_frame + 1
            for frame_id, _score in frames:
                try:
                    frame_int = int(float(str(frame_id)))
                except (TypeError, ValueError):
                    continue
                if frame_int > current_min_frame:
                    selected = frame_int
                    break
            best_frames.append(selected)
            current_min_frame = selected
        return {"video_id": video_id, "frame_ids": best_frames, "events": events, "score": None}

    def _align_video_dp(
        self,
        events: list[str],
        event_vectors: list[list[float]],
        video_id: str,
        video_stats: dict,
    ) -> dict | None:
        frames = self.db.get_frames_by_video(video_id, with_vectors=True)
        frames = [frame for frame in frames if frame.get("vector")]
        n_events = len(events)
        n_frames = len(frames)
        if n_events == 0 or n_frames < n_events:
            return self._fallback_from_hits(video_id, events, video_stats)

        sim = [
            [self._dot(event_vectors[i], frames[j]["vector"]) for j in range(n_frames)]
            for i in range(n_events)
        ]

        dp = [[-inf] * n_frames for _ in range(n_events)]
        back = [[-1] * n_frames for _ in range(n_events)]
        for j in range(n_frames):
            dp[0][j] = sim[0][j]

        for i in range(1, n_events):
            best_prev_score = -inf
            best_prev_idx = -1
            for j in range(n_frames):
                if j > 0 and dp[i - 1][j - 1] > best_prev_score:
                    best_prev_score = dp[i - 1][j - 1]
                    best_prev_idx = j - 1
                if best_prev_idx >= 0:
                    dp[i][j] = best_prev_score + sim[i][j]
                    back[i][j] = best_prev_idx

        last_idx = max(range(n_frames), key=lambda j: dp[-1][j])
        if dp[-1][last_idx] == -inf:
            return self._fallback_from_hits(video_id, events, video_stats)

        selected_indices = [last_idx]
        cur = last_idx
        for i in range(n_events - 1, 0, -1):
            cur = back[i][cur]
            selected_indices.append(cur)
        selected_indices.reverse()

        frame_ids = [frames[idx]["frame_id"] for idx in selected_indices]
        return {
            "video_id": video_id,
            "frame_ids": frame_ids,
            "events": events,
            "score": dp[-1][last_idx],
        }

    def align_events(
        self,
        events: list[str],
        top_k_per_event: int = 150,
        top_k_results: int = 1,
        candidate_videos: int = 10,
    ) -> list[dict]:
        """Zero-shot TRAKE: retrieve candidate videos, then align events with DP.

        Khong can train/nhan. Chi can SigLIP2 event vectors va frame vectors trong Qdrant.
        """
        if not events:
            return []

        event_vectors = self.encoder.encode_many(events)
        event_search_results = []
        for event in events:
            hits = self.task1.find_event(query_description=event, top_k=top_k_per_event, extract=False)
            event_search_results.append(hits)

        video_stats = defaultdict(lambda: {"match_count": 0, "total_score": 0.0, "events": defaultdict(list)})
        for event_idx, hits in enumerate(event_search_results):
            seen_videos = set()
            for hit in hits:
                vid = hit.get("video_id")
                if not vid:
                    continue
                score = float(hit.get("score") or 0.0)
                frame_id = hit.get("frame_id")
                if vid not in seen_videos:
                    video_stats[vid]["match_count"] += 1
                    seen_videos.add(vid)
                video_stats[vid]["total_score"] += score
                video_stats[vid]["events"][event_idx].append((frame_id, score))

        if not video_stats:
            return [{"error": "Khong tim thay video nao phu hop voi cac su kien."}]

        sorted_videos = sorted(
            video_stats.keys(),
            key=lambda v: (video_stats[v]["match_count"], video_stats[v]["total_score"]),
            reverse=True,
        )[:candidate_videos]

        aligned = []
        for video_id in sorted_videos:
            try:
                result = self._align_video_dp(events, event_vectors, video_id, video_stats)
                if result:
                    aligned.append(result)
            except Exception as exc:
                logger.warning("Loi align video %s bang DP, dung fallback: %s", video_id, exc)
                aligned.append(self._fallback_from_hits(video_id, events, video_stats))

        aligned.sort(
            key=lambda item: item.get("score") if item.get("score") is not None else -inf,
            reverse=True,
        )
        return aligned[:top_k_results]

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Task3TRAKEService:
    def __init__(self, task1_service):
        self.task1 = task1_service

    def align_events(self, events: list[str], top_k_per_event: int = 150, top_k_results: int = 1) -> list[dict]:
        """
        Giai đoạn 1: Truy xuất video chung cho chuỗi sự kiện.
        Giai đoạn 2: Căn chỉnh (chọn khung hình) cho từng sự kiện trong video đó, đảm bảo tính tuần tự.
        Trả về list chứa top K kết quả.
        """
        if not events:
            return []

        # 1. Tìm kiếm độc lập cho từng event
        event_search_results = []
        for event in events:
            hits = self.task1.find_event(query_description=event, top_k=top_k_per_event)
            event_search_results.append(hits)

        # 2. Chấm điểm để chọn video tốt nhất (Giai đoạn 1 - Retrieval)
        video_stats = defaultdict(lambda: {"match_count": 0, "total_score": 0.0, "events": defaultdict(list)})
        
        for event_idx, hits in enumerate(event_search_results):
            seen_videos_in_this_event = set()
            for hit in hits:
                vid = hit["video_id"]
                score = hit.get("score", 0.0)
                frame_id = hit["frame_id"]
                
                if vid not in seen_videos_in_this_event:
                    video_stats[vid]["match_count"] += 1
                    seen_videos_in_this_event.add(vid)
                
                video_stats[vid]["total_score"] += score
                video_stats[vid]["events"][event_idx].append((frame_id, score))

        if not video_stats:
            return [{"error": "Không tìm thấy video nào phù hợp với các sự kiện."}]

        # Sắp xếp để lấy Top K video có điểm cao nhất
        sorted_videos = sorted(
            video_stats.keys(),
            key=lambda v: (video_stats[v]["match_count"], video_stats[v]["total_score"]),
            reverse=True
        )
        best_videos = sorted_videos[:top_k_results]

        final_results = []

        # 3. Chọn frame cho từng video lọt top (Giai đoạn 2 - Alignment)
        for best_video in best_videos:
            best_frames = []
            current_min_frame = -1 
            missing_flags = set()
            
            for event_idx in range(len(events)):
                frames = video_stats[best_video]["events"].get(event_idx, [])
                
                if not frames:
                    fallback_frame = current_min_frame + 1
                    best_frames.append(fallback_frame)
                    current_min_frame = fallback_frame
                    missing_flags.add(event_idx)
                    continue
                
                frames.sort(key=lambda x: x[1], reverse=True)
                
                selected_frame = frames[0][0]
                found_valid = False
                for f_id, score in frames:
                    if f_id > current_min_frame:
                        selected_frame = f_id
                        found_valid = True
                        break
                
                if not found_valid:
                    missing_flags.add(event_idx)
                
                best_frames.append(selected_frame)
                current_min_frame = max(current_min_frame, selected_frame)

            # 4. Hậu xử lý: Sửa các frame có cờ thành trung bình cộng
            interpolated_frames = list(best_frames)
            for idx in sorted(list(missing_flags)):
                prev_val = None
                prev_idx = -1
                for i in range(idx - 1, -1, -1):
                    if i not in missing_flags:
                        prev_val = best_frames[i]
                        prev_idx = i
                        break
                
                next_val = None
                next_idx = -1
                for i in range(idx + 1, len(best_frames)):
                    if i not in missing_flags:
                        next_val = best_frames[i]
                        next_idx = i
                        break
                
                if prev_val is not None and next_val is not None:
                    step = (next_val - prev_val) / (next_idx - prev_idx)
                    interpolated_frames[idx] = int(prev_val + step * (idx - prev_idx))
                elif prev_val is not None:
                    interpolated_frames[idx] = prev_val + 25
                elif next_val is not None:
                    interpolated_frames[idx] = max(0, next_val - 25)

            final_results.append({
                "video_id": best_video,
                "frame_ids": interpolated_frames,
                "events": events
            })

        return final_results

from modules.task1_kis import Task1KISService


class Task3TRAKEService:
    """
    Truy vấn dạng 3: Truy xuất và căn chỉnh sự kiện video theo thời gian (TRAKE)
    Giai đoạn 1 (Retrieval): Tìm 1 video khớp nhất chứa toàn bộ chuỗi sự kiện.
    Giai đoạn 2 (Alignment): Xác định chính xác 1 semantic keyframe cho mỗi sự kiện theo thứ tự thời gian.
    """

    def __init__(self, task1_service: Task1KISService):
        self.task1 = task1_service

    def align_events(
        self, event_descriptions: list[str], top_k_per_event: int = 30
    ) -> dict:
        if not event_descriptions:
            return {}

        num_events = len(event_descriptions)
        # 1. Truy xuất danh sách ứng viên cho từng event
        event_candidates = []
        for desc in event_descriptions:
            cands = self.task1.find_event(query_description=desc, top_k=top_k_per_event)
            event_candidates.append(cands)

        # 2. Nhóm các ứng viên theo video_id
        # video_map[video_id][event_idx] = [list các hit_item]
        video_map = {}
        for ev_idx, cands in enumerate(event_candidates):
            for item in cands:
                v_id = item.get("video_id")
                if not v_id:
                    continue
                if v_id not in video_map:
                    video_map[v_id] = {i: [] for i in range(num_events)}
                video_map[v_id][ev_idx].append(item)

        best_video_id = None
        best_sequence = None
        best_total_score = -1.0

        # 3. Với mỗi video, dùng Dynamic Programming để tìm chuỗi frame tăng dần về mặt thời gian (f_0 < f_1 < ... < f_N-1)
        for v_id, events_dict in video_map.items():
            # Kiểm tra xem video này có đủ ứng viên cho tất cả các sự kiện hay không
            if any(len(events_dict[ev_idx]) == 0 for ev_idx in range(num_events)):
                continue

            # Tìm chuỗi frame_id thỏa mãn f_0 < f_1 < ... < f_{N-1} có tổng điểm score lớn nhất
            # dp[ev_idx][cand_idx] = (max_score, parent_cand_idx)
            dp = []
            for ev_idx in range(num_events):
                items = sorted(events_dict[ev_idx], key=lambda x: int(x.get("frame_id", 0)))
                events_dict[ev_idx] = items
                dp.append([-1.0] * len(items))

            # Khởi tạo cho event_idx = 0
            for c_idx, item in enumerate(events_dict[0]):
                dp[0][c_idx] = item.get("score", 0.0)

            parent = [[-1] * len(events_dict[i]) for i in range(num_events)]

            for ev_idx in range(1, num_events):
                curr_items = events_dict[ev_idx]
                prev_items = events_dict[ev_idx - 1]

                for curr_c_idx, curr_item in enumerate(curr_items):
                    curr_fid = int(curr_item.get("frame_id", 0))
                    max_prev_score = -1.0
                    best_prev_idx = -1

                    for prev_c_idx, prev_item in enumerate(prev_items):
                        prev_fid = int(prev_item.get("frame_id", 0))
                        # Ràng buộc thời gian: frame_id của sự kiện sau phải lớn hơn sự kiện trước
                        if prev_fid < curr_fid and dp[ev_idx - 1][prev_c_idx] > max_prev_score:
                            max_prev_score = dp[ev_idx - 1][prev_c_idx]
                            best_prev_idx = prev_c_idx

                    if best_prev_idx != -1:
                        dp[ev_idx][curr_c_idx] = max_prev_score + curr_item.get("score", 0.0)
                        parent[ev_idx][curr_c_idx] = best_prev_idx

            # Tìm điểm tổng lớn nhất ở event cuối cùng
            last_ev_scores = dp[num_events - 1]
            max_final_score = max(last_ev_scores) if last_ev_scores else -1.0

            if max_final_score > best_total_score:
                best_total_score = max_final_score
                best_video_id = v_id

                # Truy vết chuỗi frame
                best_last_idx = last_ev_scores.index(max_final_score)
                seq = [None] * num_events
                curr_idx = best_last_idx

                for ev_idx in range(num_events - 1, -1, -1):
                    seq[ev_idx] = events_dict[ev_idx][curr_idx]
                    curr_idx = parent[ev_idx][curr_idx]

                best_sequence = seq

        if not best_video_id or not best_sequence:
            return {}

        formatted_events = []
        for ev_idx, hit in enumerate(best_sequence):
            formatted_events.append(
                {
                    "event_index": ev_idx + 1,
                    "description": event_descriptions[ev_idx],
                    "frame_id": hit.get("frame_id"),
                    "score": hit.get("score"),
                    "image_path": hit.get("image_path"),
                    "objects_path": hit.get("objects_path"),
                    "object_classes": hit.get("object_classes", []),
                }
            )

        return {
            "video_id": best_video_id,
            "total_score": round(best_total_score, 4),
            "events": formatted_events,
        }

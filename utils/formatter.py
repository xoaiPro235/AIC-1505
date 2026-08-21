import csv
import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ==========================================
# 1. HÀM CHUẨN HÓA DỮ LIỆU (SANITIZATION)
# ==========================================


def clean_video_id(video_name: Any) -> str:
    """Chuẩn hóa tên video:
    - Loại bỏ đường dẫn thư mục (ví dụ: 'videos/L01_V028.mp4' -> 'L01_V028')
    - Loại bỏ phần mở rộng file (ví dụ: '.mp4', '.avi', '.mkv', '.mov')
    - Loại bỏ khoảng trắng thừa hai đầu
    """
    if not video_name:
        return ""
    name_str = str(video_name).strip()

    # Lấy tên file gốc nếu truyền vào đường dẫn
    name_str = Path(name_str).name

    # Xóa các extension video phổ biến
    name_str = re.sub(
        r"\.(mp4|avi|mkv|mov|webm|flv|wmv|m4v)$", "", name_str, flags=re.IGNORECASE
    )
    return name_str.strip()


def clean_frame_id(frame_id: Any, frame_offset: int = 1) -> int:
    """Chuẩn hóa Frame ID:
    - Chuyển về số nguyên int
    - Xử lý nếu frame_id có dạng '0105.jpg' hoặc '0105' hoặc float 105.0 -> 105
    - Tự động cộng thêm frame_offset (mặc định +1 vì BTC tính frame bắt đầu từ 1, còn DB/mã thường là 0-indexed)
    """
    if frame_id is None:
        return 0 + frame_offset

    frame_str = str(frame_id).strip()

    # Nếu có đuôi .jpg, .png... thì bỏ đuôi
    frame_str = re.sub(r"\.(jpg|jpeg|png|webp)$", "", frame_str, flags=re.IGNORECASE)

    try:
        # Xử lý cả float string như '123.0'
        base_id = int(float(frame_str))
    except (ValueError, TypeError):
        # Trích xuất các chữ số nếu có
        digits = re.findall(r"\d+", frame_str)
        if digits:
            base_id = int(digits[0])
        else:
            base_id = 0

    return base_id + frame_offset


def clean_qa_answer(answer: Any, max_length: int = 100) -> str:
    """Chuẩn hóa Answer cho câu hỏi Q&A:
    - Bỏ khoảng trắng thừa ở 2 đầu
    - Giới hạn tối đa `max_length` ký tự (BTC quy định tối đa 100 ký tự)
    """
    if answer is None:
        return ""
    ans_str = str(answer).strip()

    # Bỏ dấu ngoặc kép thừa bọc ngoài nếu model sinh ra dạng '"5"'
    if (ans_str.startswith('"') and ans_str.endswith('"')) or (
        ans_str.startswith("'") and ans_str.endswith("'")
    ):
        ans_str = ans_str[1:-1].strip()

    # Giới hạn độ dài tối đa 100 ký tự theo yêu cầu BTC
    if len(ans_str) > max_length:
        logger.warning(
            "Answer dài hơn %d ký tự (%d ký tự). Đang tự động cắt ngắn.",
            max_length,
            len(ans_str),
        )
        ans_str = ans_str[:max_length].strip()

    return ans_str


# ==========================================
# 2. HÀM FORMAT TỪNG DÒNG (ROW FORMATTERS)
# ==========================================


def format_kis_row(
    video_id: Any, frame_id: Any, frame_offset: int = 1
) -> list[str]:
    """Format 1 dòng cho Task 1 (Textual KIS):
    Format: <video_name>, <frame_id>
    Ví dụ: ['L00_V000', '1234']
    """
    v_id = clean_video_id(video_id)
    f_id = clean_frame_id(frame_id, frame_offset=frame_offset)
    return [v_id, str(f_id)]


def format_qa_row(
    video_id: Any, frame_id: Any, answer: Any, frame_offset: int = 1
) -> list[str]:
    """Format 1 dòng cho Task 2 (Q&A):
    Format: <video_name>, <frame_id>, <answer>
    Ví dụ: ['L01_V028', '3450', '5']
    """
    v_id = clean_video_id(video_id)
    f_id = clean_frame_id(frame_id, frame_offset=frame_offset)
    ans = clean_qa_answer(answer)
    return [v_id, str(f_id), ans]


def format_trake_row(
    video_id: Any, frame_ids: list[Any], frame_offset: int = 1
) -> list[str]:
    """Format 1 dòng cho Task 3 (TRAKE):
    Format: <video_name>, <frame_1>, <frame_2>, ..., <frame_N>
    Ví dụ: ['L10_V001', '1200', '1850', '2100', '2450']
    """
    v_id = clean_video_id(video_id)
    clean_fids = [
        str(clean_frame_id(fid, frame_offset=frame_offset)) for fid in frame_ids
    ]
    return [v_id] + clean_fids


# ==========================================
# 3. HÀM XUẤT FILE CSV CHO TỪNG LOẠI TASK
# ==========================================


def export_kis_csv(
    results: list[dict] | list[tuple] | list[list] | None,
    output_path: str | Path,
    max_rows: int = 100,
    frame_offset: int = 1,
) -> Path:
    """Xuất file CSV cho kết quả Textual KIS (Task 1).
    - Hỗ trợ kết quả dạng list dict từ task1.find_event() hoặc list of tuples (video_id, frame_id).
    - Tối đa 100 dòng theo quy định BTC.
    - Không có header, delimiter dấu phẩy, UTF-8.
    - frame_offset mặc định là 1 (chuyển 0-indexed sang 1-indexed theo quy định của BTC).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    items_list = results if results is not None else []
    for item in items_list[:max_rows]:
        if isinstance(item, dict):
            v_id = item.get("video_id") or item.get("video_name") or item.get("video")
            f_id = item.get("frame_id") or item.get("frame_idx") or item.get("frame")
            rows.append(format_kis_row(v_id, f_id, frame_offset=frame_offset))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append(format_kis_row(item[0], item[1], frame_offset=frame_offset))

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

    logger.info(
        "Đã xuất thành công %d dòng KIS ra file: %s (offset=%d)", len(rows), out_p.resolve(), frame_offset
    )
    return out_p


def export_qa_csv(
    results: list[dict] | list[tuple] | list[list] | None,
    output_path: str | Path,
    max_rows: int = 100,
    frame_offset: int = 1,
) -> Path:
    """Xuất file CSV cho kết quả Q&A (Task 2).
    - Hỗ trợ kết quả dạng list dict từ task2.answer_question() / task2.qa_search() hoặc list of tuples (video_id, frame_id, answer).
    - Tối đa 100 dòng.
    - Tự động bao dấu ngoặc kép an toàn cho trường answer nếu có dấu phẩy hoặc ký tự đặc biệt.
    - frame_offset mặc định là 1 (1-indexed theo quy định của BTC).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    items_list = results if results is not None else []
    for item in items_list[:max_rows]:
        if isinstance(item, dict):
            v_id = item.get("video_id") or item.get("video_name") or item.get("video")
            f_id = item.get("frame_id") or item.get("frame_idx") or item.get("frame")
            ans = item.get("answer", "")
            rows.append(format_qa_row(v_id, f_id, ans, frame_offset=frame_offset))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            rows.append(format_qa_row(item[0], item[1], item[2], frame_offset=frame_offset))

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        # csv.QUOTE_MINIMAL tự động quote khi có dấu phẩy, quote kép, newline
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

    logger.info(
        "Đã xuất thành công %d dòng QA ra file: %s (offset=%d)", len(rows), out_p.resolve(), frame_offset
    )
    return out_p


def export_trake_csv(
    results: list[dict] | dict | list[tuple] | list[list] | None,
    output_path: str | Path,
    expected_events_count: int | None = None,
    max_rows: int = 100,
    frame_offset: int = 1,
) -> Path:
    """Xuất file CSV cho kết quả TRAKE (Task 3).
    - Hỗ trợ kết quả từ task3.align_events() (dạng dict đơn lẻ hoặc list các predictions).
    - Mỗi dòng: <video_name>, <frame_1>, <frame_2>, ..., <frame_N>
    - Kiểm tra số lượng frame_ids phải khớp với số events yêu cầu.
    - frame_offset mặc định là 1 (1-indexed theo quy định của BTC).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Chuẩn hóa results thành list
    if results is None:
        items_list = []
    elif isinstance(results, dict):
        if "predictions" in results and isinstance(results["predictions"], list):
            items_list = results["predictions"]
        else:
            items_list = [results]
    else:
        items_list = list(results)

    rows = []
    for item in items_list[:max_rows]:
        if isinstance(item, dict):
            # Bỏ qua nếu là thông báo lỗi
            if "error" in item and "video_id" not in item:
                logger.warning("Bỏ qua kết quả có lỗi: %s", item.get("error"))
                continue

            v_id = item.get("video_id") or item.get("video_name") or item.get("video")
            f_ids = []

            # 1. Ưu tiên lấy từ frame_ids (dạng list các integer)
            if "frame_ids" in item and isinstance(item["frame_ids"], list):
                f_ids = item["frame_ids"]
            # 2. Nếu không có frame_ids, trích xuất từ events (dạng list dict)
            elif "events" in item and isinstance(item["events"], list):
                f_ids = [
                    ev.get("frame_id")
                    for ev in item["events"]
                    if isinstance(ev, dict) and ev.get("frame_id") is not None
                ]

            if v_id and f_ids:
                if (
                    expected_events_count is not None
                    and len(f_ids) != expected_events_count
                ):
                    logger.warning(
                        "Số frame của video %s (%d frames) không khớp với expected_events_count (%d)",
                        v_id,
                        len(f_ids),
                        expected_events_count,
                    )
                rows.append(format_trake_row(v_id, f_ids, frame_offset=frame_offset))

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            v_id = item[0]
            f_ids = item[1:]
            rows.append(format_trake_row(v_id, f_ids, frame_offset=frame_offset))

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

    logger.info(
        "Đã xuất thành công %d dòng TRAKE ra file: %s (offset=%d)", len(rows), out_p.resolve(), frame_offset
    )
    return out_p


# ==========================================
# 4. HÀM ĐỌC VÀ NHẬN DIỆN QUERY CỦA BTC
# ==========================================


def parse_query_file(query_file_path: str | Path) -> dict[str, Any]:
    """Đọc file query dạng text (.txt) từ ban tổ chức và tự động nhận diện loại truy vấn:
    - Hậu tố '-kis.txt' -> 'kis'
    - Hậu tố '-qa.txt' -> 'qa'
    - Hậu tố '-trake.txt' -> 'trake'

    Trả về dict:
    {
        "query_id": "query-1-kis",
        "query_type": "kis" | "qa" | "trake",
        "raw_text": "...",
        "kis_description": "...",        (nếu là KIS)
        "qa_question": "...",            (nếu là QA)
        "trake_events": ["event 1", ...] (nếu là TRAKE)
        "output_csv_name": "query-1-kis.csv"
    }
    """
    file_p = Path(query_file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file query: {file_p}")

    filename = file_p.name
    stem = file_p.stem  # ví dụ 'query-1-kis'

    content = file_p.read_text(encoding="utf-8").strip()

    # Nhận diện loại task từ tên file
    lower_stem = stem.lower()
    if lower_stem.endswith("-kis") or "_kis" in lower_stem:
        q_type = "kis"
    elif lower_stem.endswith("-qa") or "_qa" in lower_stem:
        q_type = "qa"
    elif lower_stem.endswith("-trake") or "_trake" in lower_stem:
        q_type = "trake"
    else:
        # Fallback nhận diện theo cấu trúc nội dung
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) > 1:
            q_type = "trake"
        elif "?" in content or "gì" in content.lower() or "ai" in content.lower():
            q_type = "qa"
        else:
            q_type = "kis"

    result: dict[str, Any] = {
        "query_id": stem,
        "query_type": q_type,
        "raw_text": content,
        "output_csv_name": f"{stem}.csv",
    }

    if q_type == "kis":
        result["kis_description"] = content
    elif q_type == "qa":
        result["qa_question"] = content
    elif q_type == "trake":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        event_pattern = re.compile(
            r"^(?:E\s*\d+|Event\s*\d+|Sự\s*kiện\s*\d+|Su\s*kien\s*\d+)[\.\)\-:]\s*(.+)$",
            flags=re.IGNORECASE,
        )

        context_lines = []
        clean_events = []
        saw_labeled_event = False

        for line in lines:
            match = event_pattern.match(line)
            if match:
                saw_labeled_event = True
                event_text = match.group(1).strip()
                if event_text:
                    clean_events.append(event_text)
                continue

            if saw_labeled_event:
                # Nếu event bị wrap xuống dòng tiếp theo, nối vào event gần nhất.
                if clean_events:
                    clean_events[-1] = f"{clean_events[-1]} {line}".strip()
            else:
                context_lines.append(line.rstrip(":"))

        if not clean_events:
            # Fallback cũ: mỗi dòng là một event, có lọc số thứ tự đầu dòng.
            for ev in lines:
                ev_clean = re.sub(
                    r"^(\d+[\.\)\-:]|\bEvent\s*\d+[\.\)\-:]?)\s*",
                    "",
                    ev,
                    flags=re.IGNORECASE,
                ).strip()
                clean_events.append(ev_clean if ev_clean else ev)

        trake_context = " ".join(context_lines).strip()
        if trake_context and clean_events:
            context_prefix = trake_context.rstrip(".。")
            clean_events = [f"{context_prefix}. {event}" for event in clean_events]

        result["trake_context"] = trake_context
        result["trake_events"] = clean_events

    return result


# ==========================================
# 5. ĐÓNG GÓI SUBMISSION (.ZIP)
# ==========================================


def create_submission_zip(
    submission_dir: str | Path = "submission",
    output_zip_path: str | Path = "submission.zip",
) -> Path:
    """Nén thư mục `submission/` thành file `.zip`.
    QUY ĐỊNH BẮT BUỘC CỦA BTC:
    - Trong file zip BẮT BUỘC phải chứa thư mục 'submission/' ở root (không nén trực tiếp file CSV rời).
    - Ví dụ:
        submission.zip
        └── submission/
            ├── query-1-kis.csv
            ├── query-2-kis.csv
            ├── query-3-qa.csv
            └── query-4-trake.csv
    """
    sub_dir = Path(submission_dir)
    if not sub_dir.exists() or not sub_dir.is_dir():
        raise FileNotFoundError(f"Thư mục submission không tồn tại: {sub_dir}")

    zip_p = Path(output_zip_path)
    zip_p.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(sub_dir.glob("*.csv")))
    if not csv_files:
        logger.warning(
            "CẢNH BÁO: Thư mục %s không có bất kỳ file .csv nào!", sub_dir
        )

    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zipf:
        for csv_file in csv_files:
            # Đường dẫn tương đối bên trong zip: 'submission/query-x.csv'
            arcname = f"{sub_dir.name}/{csv_file.name}"
            zipf.write(csv_file, arcname=arcname)

    logger.info("Đã tạo file zip submission thành công tại: %s", zip_p.resolve())
    return zip_p


# ==========================================
# 6. BỘ KIỂM TRA ĐỊNH DẠNG (VALIDATOR & CHECKLIST)
# ==========================================


def validate_submission(submission_dir: str | Path = "submission") -> dict[str, Any]:
    """Kiểm tra toàn diện tất cả các file CSV trong thư mục submission theo Checklist của BTC.

    Kiểm tra:
    1. Đuôi file là .csv
    2. Encoding UTF-8 hợp lệ
    3. Không có dòng Header
    4. Số dòng trong khoảng 1 đến 100
    5. Tên video KHÔNG chứa đuôi (.mp4, .avi, ...)
    6. Frame ID là số nguyên hợp lệ
    7. Q&A: Answer <= 100 ký tự
    8. TRAKE: Đúng định dạng chuỗi các frame ID tăng dần

    Trả về dict chi tiết kết quả kiểm tra cho từng file.
    """
    sub_dir = Path(submission_dir)
    report: dict[str, Any] = {
        "submission_dir": str(sub_dir),
        "is_valid": True,
        "files_count": 0,
        "files_report": {},
        "errors": [],
        "warnings": [],
    }

    if not sub_dir.exists() or not sub_dir.is_dir():
        report["is_valid"] = False
        report["errors"].append(
            f"Thư mục submission không tồn tại: {sub_dir.resolve()}"
        )
        return report

    csv_files = sorted(list(sub_dir.glob("*.csv")))
    report["files_count"] = len(csv_files)

    if not csv_files:
        report["is_valid"] = False
        report["errors"].append(f"Không tìm thấy file .csv nào trong {sub_dir}")
        return report

    print("\n" + "=" * 65)
    print("      KIỂM TRA ĐỊNH DẠNG SUBMISSION TRƯỚC KHI NỘP BÀI")
    print("=" * 65)

    all_passed = True

    for csv_path in csv_files:
        file_errors = []
        file_warnings = []
        file_stem = csv_path.stem.lower()

        # Nhận diện loại task
        if "kis" in file_stem:
            expected_type = "kis"
        elif "qa" in file_stem:
            expected_type = "qa"
        elif "trake" in file_stem:
            expected_type = "trake"
        else:
            expected_type = "unknown"

        # Đọc file CSV
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = list(csv.reader(f))
        except UnicodeDecodeError:
            file_errors.append("File không đúng chuẩn mã hóa UTF-8!")
            reader = []
        except Exception as e:
            file_errors.append(f"Lỗi đọc file: {e}")
            reader = []

        total_rows = len(reader)
        if total_rows == 0:
            file_errors.append("File rỗng (0 dòng)!")
        elif total_rows > 100:
            file_errors.append(
                f"File vượt quá 100 dòng cho phép (hiện có {total_rows} dòng)!"
            )

        # Kiểm tra từng dòng
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(col.strip() == "" for col in row):
                file_warnings.append(f"Dòng {row_idx}: Dòng trống.")
                continue

            video_id = row[0].strip()

            # Kiểm tra lỗi có đuôi video .mp4
            if re.search(r"\.(mp4|avi|mkv|mov)$", video_id, flags=re.IGNORECASE):
                file_errors.append(
                    f"Dòng {row_idx}: Tên video '{video_id}' còn chứa đuôi mở rộng (.mp4/.avi)!"
                )

            # Kiểm tra dòng đầu có phải Header không (ví dụ chứa chữ 'video', 'frame', 'answer')
            if row_idx == 1:
                col_headers = [c.lower().strip() for c in row]
                if any(
                    h in ["video", "video_name", "frame", "frame_id", "answer", "score"]
                    for h in col_headers
                ):
                    file_errors.append(
                        "Dòng 1 có vẻ là HEADER! BTC yêu cầu KHÔNG CÓ header trong file CSV."
                    )

            if expected_type == "kis":
                if len(row) < 2:
                    file_errors.append(
                        f"Dòng {row_idx}: KIS cần tối thiểu 2 cột (<video>, <frame_id>), hiện có {len(row)} cột."
                    )
                else:
                    fid_str = row[1].strip()
                    if not fid_str.lstrip("-").isdigit():
                        file_errors.append(
                            f"Dòng {row_idx}: Frame ID '{fid_str}' không phải số nguyên!"
                        )

            elif expected_type == "qa":
                if len(row) < 3:
                    file_errors.append(
                        f"Dòng {row_idx}: Q&A cần 3 cột (<video>, <frame_id>, <answer>), hiện có {len(row)} cột."
                    )
                else:
                    fid_str = row[1].strip()
                    if not fid_str.lstrip("-").isdigit():
                        file_errors.append(
                            f"Dòng {row_idx}: Frame ID '{fid_str}' không phải số nguyên!"
                        )
                    answer_str = row[2]
                    if len(answer_str) > 100:
                        file_errors.append(
                            f"Dòng {row_idx}: Answer dài {len(answer_str)} ký tự (quá giới hạn 100 ký tự)!"
                        )

            elif expected_type == "trake":
                if len(row) < 3:
                    file_errors.append(
                        f"Dòng {row_idx}: TRAKE cần ít nhất 1 video + 2 events trở lên, hiện có {len(row)} cột."
                    )
                else:
                    fids = []
                    for col_i, fid_str in enumerate(row[1:], start=1):
                        clean_f = fid_str.strip()
                        if not clean_f.lstrip("-").isdigit():
                            file_errors.append(
                                f"Dòng {row_idx}, Cột {col_i + 1}: Frame ID '{clean_f}' không phải số nguyên!"
                            )
                        else:
                            fids.append(int(clean_f))

                    # Kiểm tra thứ tự thời gian tăng dần
                    if len(fids) > 1 and fids != sorted(fids):
                        file_warnings.append(
                            f"Dòng {row_idx}: Chuỗi frame {fids} không tăng dần theo thời gian!"
                        )

        is_file_valid = len(file_errors) == 0
        if not is_file_valid:
            all_passed = False

        status_icon = "✅ HỢP LỆ" if is_file_valid else "❌ CÓ LỖI"
        print(f"\n📄 File: {csv_path.name} [{status_icon}]")
        print(f"   - Loại task nhận diện: {expected_type.upper()}")
        print(f"   - Số dòng: {total_rows}/100")
        if total_rows > 0 and len(reader) > 0:
            preview_row = ", ".join(reader[0][:4])
            if len(reader[0]) > 4:
                preview_row += ", ..."
            print(f"   - Ví dụ dòng 1: {preview_row}")

        if file_errors:
            for err in file_errors:
                print(f"     🔴 LỖI: {err}")
                report["errors"].append(f"{csv_path.name}: {err}")
        if file_warnings:
            for warn in file_warnings:
                print(f"     ⚠️  CẢNH BÁO: {warn}")
                report["warnings"].append(f"{csv_path.name}: {warn}")

        report["files_report"][csv_path.name] = {
            "is_valid": is_file_valid,
            "type": expected_type,
            "rows": total_rows,
            "errors": file_errors,
            "warnings": file_warnings,
        }

    report["is_valid"] = all_passed
    print("\n" + "=" * 65)
    if all_passed:
        print("🎉 TẤT CẢ FILE SUBMISSION ĐÃ HỢP LỆ THEO CHUẨN BTC!")
    else:
        print("❌ CÓ FILE CHƯA HỢP LỆ. VUI LÒNG KIỂM TRA LẠI CÁC LỖI TRÊN!")
    print("=" * 65 + "\n")

    return report

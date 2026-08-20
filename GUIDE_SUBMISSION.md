# Hướng Dẫn Sử Dụng Module Submission & Format Chuẩn BTC AIC

Module `utils/formatter.py` và `pipeline.py` đã được thiết kế đầy đủ để phục vụ cho buổi thi vòng sơ tuyển AIC với 3 dạng bài: **Textual KIS**, **Q&A**, và **TRAKE**.

---

## 1. Tóm Tắt Quy Định Định Dạng Của BTC

| Dạng Bài | Tên File Query Mẫu | Tên File CSV Nộp | Định Dạng Dòng Kết Quả | Ví Dụ Dòng |
| :--- | :--- | :--- | :--- | :--- |
| **1. Textual KIS** | `query-1-kis.txt` | `query-1-kis.csv` | `<video_name>, <frame_id>` | `L01_V028, 25300` |
| **2. Q&A** | `query-2-qa.txt` | `query-2-qa.csv` | `<video_name>, <frame_id>, "<answer>"` | `L01_V028, 3450, "5"` |
| **3. TRAKE** | `query-3-trake.txt` | `query-3-trake.csv` | `<video_name>, <frame_1>, <frame_2>, ...` | `L10_V001, 1200, 1850, 2100` |

### ⚠️ Lưu ý sống còn:
1. **Không có Header**: Không có dòng tiêu đề cột (`video,frame,...`).
2. **Tên video KHÔNG có đuôi file**: Dùng `L01_V028`, KHÔNG dùng `L01_V028.mp4`.
3. **Frame ID**: Luôn là số nguyên (`25300`).
4. **Answer (Q&A)**: Tối đa 100 ký tự. Có dấu phẩy hoặc ký tự đặc biệt sẽ tự động được escape bằng dấu ngoặc kép.
5. **Số dòng**: Tối đa 100 dòng cho mỗi file CSV.
6. **Đóng gói ZIP**: File nén `.zip` BẮT BUỘC phải chứa thư mục `submission/` ở root:
   ```
   submission.zip
   └── submission/
       ├── query-1-kis.csv
       ├── query-2-kis.csv
       ├── query-3-qa.csv
       └── query-4-trake.csv
   ```

---

## 2. Cách Sử Dụng Trong Code Python

Bạn có thể import các hàm từ `utils.formatter`:

```python
from utils.formatter import (
    clean_video_id,
    clean_frame_id,
    clean_qa_answer,
    export_kis_csv,
    export_qa_csv,
    export_trake_csv,
    validate_submission,
    create_submission_zip,
)

# 1. Xuất file Textual KIS
# results có dạng list[dict] từ task1.find_event(...)
export_kis_csv(res_task1, "submission/query-1-kis.csv", max_rows=100)

# 2. Xuất file Q&A
# results có dạng list[dict] từ task2.answer_question(...)
export_qa_csv(res_task2, "submission/query-2-qa.csv", max_rows=100)

# 3. Xuất file TRAKE
# res_task3 là kết quả từ task3.align_events(...)
export_trake_csv(res_task3, "submission/query-3-trake.csv", max_rows=100)

# 4. Kiểm tra hợp lệ toàn bộ file trước khi nộp
validate_submission("submission")

# 5. Đóng gói file ZIP nộp bài
create_submission_zip("submission", "submission.zip")
```

---

## 3. Chạy Tự Động Toàn Bộ Bằng Pipeline CLI

Khi BTC cấp một gói gồm nhiều file `query-*.txt` (đặt vào thư mục `queries/`), bạn chỉ cần chạy:

```bash
python pipeline.py --queries_dir queries --zip_name submission_round1.zip
```

Hoặc chỉ kiểm tra và nén thư mục `submission/` có sẵn:

```bash
python pipeline.py --validate_only --output_dir submission --zip_name submission.zip
```

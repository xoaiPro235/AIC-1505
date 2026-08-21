# TÀI LIỆU ĐẶC TẢ KỸ THUẬT - DỰ ÁN AI CHALLENGE (AIC) 2026

## PHẦN 1: TỔNG QUAN 3 DẠNG TRUY VẤN CỐT LÕI

Hệ thống được thiết kế để giải quyết 3 bài toán từ Ban tổ chức (BTC) với độ khó tăng dần. Yêu cầu bắt buộc là độ chính xác cao và thời gian phản hồi (Latency) thấp.

### 1. Dạng 1: Textual KIS (Tìm kiếm chính xác theo văn bản)

* **Mục tiêu:** Tìm đúng 1 video và định vị chính xác 1 khung hình duy nhất khớp với đoạn mô tả sự kiện bằng ngôn ngữ tự nhiên.
* **Ví dụ truy vấn:** *"Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời."*
* **Định dạng kết quả nộp:** `<video_id>, <frame_id>` (VD: `L01_V001, 1500`).

### 2. Dạng 2: Q&A (Hỏi-Đáp trực quan)

* **Mục tiêu:** Tìm đoạn video chứa bối cảnh, cô lập vật thể mục tiêu và trích xuất câu trả lời trực tiếp cho một câu hỏi cụ thể.
* **Ví dụ truy vấn:** Bối cảnh *"Lễ trao giải âm nhạc"* - Câu hỏi: *"Có bao nhiêu người lên sân khấu nhận giải?"*
* **Định dạng kết quả nộp:** `<video_id>, <frame_id>, <answer>` (VD: `L01_V001, 3450, 5`).

### 3. Dạng 3: TRAKE (Truy xuất và căn chỉnh sự kiện theo thời gian)

* **Mục tiêu:** Nhiệm vụ phức hợp. Tìm ra video chứa chuỗi hành động và xác định chính xác $N$ khung hình ngữ nghĩa tương ứng với $N$ khoảnh khắc xảy ra liên tiếp theo đúng trình tự thời gian.
* **Ví dụ truy vấn:** Hành động "Nhảy cao" gồm 4 sự kiện: (1) Chạy đà $\rightarrow$ (2) Giậm nhảy $\rightarrow$ (3) Bay qua xà $\rightarrow$ (4) Tiếp đất.
* **Định dạng kết quả nộp:** `<video_id>, <frame_1>, <frame_2>, ..., <frame_N>` (VD: `L10_V010, 101, 156, 203, 251`).

---

## PHẦN 2: KIẾN TRÚC LƯU TRỮ TỐI ƯU (DECOUPLED STORAGE)

Để đảm bảo hệ thống vận hành trơn tru trên phần cứng giới hạn (như RTX 3060 12GB VRAM) mà không bị tràn bộ nhớ khi xử lý hàng triệu khung hình, nhóm áp dụng triết lý **Lưu trữ Phân tách đa tầng (Multi-tier Decoupled Storage)**.

### 2.1. Phân tầng Dữ liệu

* **Tầng SSD (File System):** Lưu trữ toàn bộ dữ liệu dung lượng lớn.
* Ảnh khung hình (`.jpg`).
* Tọa độ đối tượng (`.json`) trích xuất từ YOLO.


* **Tầng RAM (Vector DB - Qdrant):** Chỉ lưu trữ ma trận số học và siêu dữ liệu (Metadata) tinh gọn để tính toán siêu tốc.

### 2.2. Cấu trúc Payload trong Qdrant

Mỗi một bản ghi (Point) trong cơ sở dữ liệu Qdrant sẽ bao gồm 1 Vector 512 chiều (chuẩn hóa L2) từ CLIP và một Payload đính kèm như sau:

```json
{
    "video_id": "L01_V001",
    "frame_id": 105,
    "image_path": "/data/frames/L01_V001/0105.jpg",
    "objects_path": "/data/objects/L01_V001/0105.json",
    "object_classes": ["person", "car", "traffic light"]
}

```

*Lưu ý:* Hệ thống bắt buộc phải khởi tạo **Keyword Index** cho trường `video_id` và `object_classes` để kích hoạt tính năng Pre-filtering (Lọc thô). Khoảng cách hình học được thiết lập là `Distance.DOT` thay vì Cosine để tăng tốc độ nhân ma trận trên CPU/GPU.

---

## PHẦN 3: LUỒNG THỰC THI (EXECUTION PIPELINE)

Hệ thống hoạt động theo nguyên lý "Bộ định tuyến Tác vụ" (Task Router). Dữ liệu sẽ đi qua các pipeline khác nhau tùy thuộc vào loại truy vấn.

### 3.1. Luồng Dạng 1 (KIS) - Hybrid Search

1. **Tiền xử lý:** Trích xuất các danh từ chỉ vật thể từ câu truy vấn (VD: `["person", "car"]`). Text Encoder của CLIP chuyển câu truy vấn thành Vector $q$.
2. **Lọc Logic (Pre-filtering):** Qdrant dùng mảng `object_classes` lọc bỏ tức thời các khung hình không chứa vật thể mục tiêu.
3. **Toán học (Dense Search):** Qdrant tính phép nhân vô hướng (DOT) giữa vector $q$ và các khung hình còn lại để lấy Top-1 `video_id` và `frame_id`.
4. **Hoàn thành:** Xuất kết quả.

### 3.2. Luồng Dạng 2 (Q&A) - Visual Language Pipeline

1. **Định vị:** Chạy Luồng Dạng 1 để tìm ra Top-1 `frame_id` (bối cảnh chuẩn nhất).
2. **I/O Đĩa cứng:** Đọc `image_path` và `objects_path` từ Payload của Qdrant. Trích xuất bounding box từ file JSON và cắt (crop) ảnh cô lập vật thể.
3. **Suy luận LLM:** Nạp ảnh đã cắt và câu hỏi vào mô hình Qwen-VL-Chat (Lượng tử hóa 4-bit) để trích xuất câu trả lời ngắn gọn.
4. **Hoàn thành:** Gắn kết quả thành `<video_id>, <frame_id>, <answer>`.

### 3.3. Luồng Dạng 3 (TRAKE) - Dynamic Time Warping (DTW)

1. **Định vị Video Toàn cục:** Gộp $N$ câu truy vấn sự kiện, dùng Qdrant tìm kiếm tập hợp để khoanh vùng ra đúng 1 `video_id` duy nhất có mật độ khớp cao nhất.
2. **Tải Ma trận Cục bộ:** Dùng API của Qdrant truy vấn lại toàn bộ vector khung hình chỉ thuộc về `video_id` đó (VD: tải ma trận $V \in \mathbb{R}^{M \times 512}$).
3. **Quy hoạch Động (DTW Alignment):**
* Lập Ma trận Chi phí chéo (Cross-cost Matrix) giữa $N$ vector truy vấn và $M$ vector khung hình.
* Áp dụng phương trình Bellman để trượt mảng sự kiện dọc theo trục thời gian, đảm bảo tính đơn điệu (không đi lùi).


4. **Hoàn thành:** Truy vết (Traceback) đường dẫn có tổng chi phí Cosine nhỏ nhất để lấy ra $N$ `frame_id` tối ưu. Xuất chuỗi kết quả nộp bài.


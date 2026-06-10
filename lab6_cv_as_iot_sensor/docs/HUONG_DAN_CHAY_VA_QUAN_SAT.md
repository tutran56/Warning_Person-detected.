# Hướng dẫn chạy và quan sát Lab 6

## 1. Mục tiêu của bài lab

Lab 6 xây dựng một pipeline thị giác máy tính ở mức nhập môn cho hệ thống AIoT. Sau khi chạy, cần quan sát được camera stream, snapshot, video ngắn, motion capture, ảnh xử lý, metadata và event.

## 2. Cài môi trường

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

Quan sát: lệnh cài đặt không báo lỗi; có thể import `cv2`, `fastapi`, `PIL`.

## 3. Chạy thử không cần camera

```bash
python run_lab6_demo.py
```

Quan sát:

- `RUN_TEST_LOG.txt` có `LOCAL_PIPELINE_TEST_PASS`.
- `data/raw_images/` có ảnh mới.
- `data/processed_images/` có ảnh tổng hợp bốn bước.
- `data/videos/` có video ngắn.
- `outputs/image_metadata.csv` có dòng metadata.
- `outputs/image_event_log.csv` có event.

Ý nghĩa: pipeline hoạt động ngay cả khi chưa có camera thật.

## 4. Chạy giao diện web

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở:

```text
http://127.0.0.1:8000/
```

Thao tác cần thực hiện:

1. Nhập `0` ở ô Camera source.
2. Bấm `Bật stream`.
3. Bấm `Chụp snapshot`.
4. Bấm `Ghi video 5s`.
5. Bấm `Phát hiện chuyển động và chụp ảnh`.
6. Upload một ảnh có sẵn.
7. Quan sát ảnh gốc, ảnh xử lý, số lượng metadata, số lượng event.

## 5. Dùng IP camera

Nếu có IP camera, thay `0` bằng URL stream, ví dụ:

```text
http://192.168.1.10:8080/video
rtsp://username:password@192.168.1.20:554/stream1
```

Nếu URL không truy cập được, hệ thống chuyển sang stream mô phỏng để bài lab vẫn tiếp tục.

## 6. Kết luận bắt buộc sau khi chạy

Báo cáo ngắn cần trả lời:

1. Ảnh từ camera đi qua những bước nào?
2. Metadata giúp hệ thống biết gì về ảnh?
3. Event khác metadata như thế nào?
4. Motion capture có phải object detection không?
5. Lab 6 tạo dữ liệu đầu vào gì cho Lab 7?

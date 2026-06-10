# Phân tích code Lab 6

## 1. Vì sao chỉ dùng ít file Python?

Lab 6 chỉ có hai file Python chính để tránh vụn nội dung:

- `app.py`: chứa toàn bộ logic backend và pipeline ảnh.
- `run_lab6_demo.py`: tạo dữ liệu thử nhanh không cần mở camera thật.

Giao diện quan sát nằm trong `index.html`, giúp tách phần hiển thị khỏi phần xử lý ảnh.

## 2. Luồng chính trong `app.py`

```text
camera/upload image
→ frame BGR trong OpenCV
→ lưu ảnh gốc
→ tạo ảnh xử lý bốn bước
→ ghi image_metadata.csv
→ ghi image_event_log.csv
→ trả JSON và URL ảnh cho dashboard
```

## 3. Các hàm quan trọng

| Hàm | Vai trò |
|---|---|
| `log_image_pipeline()` | Hàm lõi: lưu ảnh, xử lý ảnh, ghi metadata và event |
| `create_processed_contact_sheet()` | Tạo ảnh quan sát gồm resize, grayscale, threshold, edge |
| `stream_frames()` | Tạo live stream MJPEG từ camera hoặc stream mô phỏng |
| `record_short_video()` | Ghi video ngắn từ camera |
| `motion_capture()` | So sánh frame để phát hiện chuyển động và chụp ảnh |

## 4. Phân tích `index.html`

`index.html` là dashboard một file. Các nút trên giao diện gọi API của `app.py`:

| Nút | API được gọi | Ý nghĩa |
|---|---|---|
| Bật stream | `/video_feed` | Xem camera đang chạy |
| Chụp snapshot | `/snapshot` | Lưu một ảnh từ camera |
| Ghi video 5s | `/record-video` | Lưu video ngắn |
| Phát hiện chuyển động | `/motion-capture` | Tự chụp ảnh khi frame thay đổi |
| Upload ảnh | `/upload-image` | Đưa ảnh có sẵn vào pipeline |

## 5. Điểm cần hiểu

- `metadata` mô tả ảnh.
- `event` mô tả điều có ý nghĩa vận hành từ ảnh.
- `motion detection` chỉ phát hiện thay đổi giữa các frame, chưa phải nhận diện vật thể.
- Lab 7 sẽ dùng ảnh từ Lab 6 để chạy object detection.

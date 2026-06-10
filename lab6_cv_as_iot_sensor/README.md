# Lab 6 - Computer Vision as IoT Sensor

Lab này đưa camera/ảnh vào hệ thống AIoT như một cảm biến trực quan. Mục tiêu là chạy được live stream, chụp ảnh, ghi video, phát hiện chuyển động, xử lý ảnh cơ bản, ghi metadata, sinh event và quan sát trên dashboard HTML.

## Cấu trúc file chính

```text
app.py              # backend FastAPI: stream, snapshot, video, motion, preprocess, metadata, event
index.html          # giao diện dashboard: stream, upload ảnh, quan sát ảnh/metadata/event
run_lab6_demo.py    # chạy thử nhanh không cần camera thật
```

## Chạy nhanh

```bash
python -m venv .venv
# Windows
.venv\Scriptsctivate
# macOS/Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
python run_lab6_demo.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

## Cần quan sát sau khi chạy

- `data/raw_images/`: ảnh gốc từ upload/snapshot/motion.
- `data/processed_images/`: ảnh tổng hợp bốn bước xử lý.
- `data/videos/`: video ngắn ghi từ camera hoặc stream mô phỏng.
- `outputs/image_metadata.csv`: metadata của ảnh.
- `outputs/image_event_log.csv`: event sinh từ ảnh/camera.
- Dashboard tại `/`: live stream, ảnh gốc, ảnh xử lý, bảng metadata và event.

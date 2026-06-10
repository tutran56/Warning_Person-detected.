
"""Run a quick Lab 6 smoke demo without opening the web server or camera."""
from pathlib import Path
import json
import cv2
from app import ROOT, VIDEO_DIR, EVENT_CSV, EVENT_FIELDS, append_csv, log_image_pipeline, simulated_frame, record_short_video, motion_capture

log_lines = []
try:
    # Process two simulated frames so metadata/event files become observable immediately.
    for i in range(2):
        frame = simulated_frame(i)
        result = log_image_pipeline(frame, source_type="demo_script", device_id="simulated_camera", note=f"demo_frame={i}")
        log_lines.append(json.dumps({"image_id": result["image_id"], "event": result["event"]["event_type"]}, ensure_ascii=False))
    video_result = record_short_video("no_camera_fallback", seconds=1)
    log_lines.append(json.dumps({"video_id": video_result["video_id"], "frames": video_result["frames"]}, ensure_ascii=False))
    motion_result = motion_capture("no_camera_fallback", seconds=1)
    log_lines.append(json.dumps({"motion_image_id": motion_result["image_id"], "motion_event": motion_result["motion_event"]["event_type"]}, ensure_ascii=False))
    status = "LOCAL_PIPELINE_TEST_PASS"
except Exception as exc:
    status = f"LOCAL_PIPELINE_TEST_FAIL: {exc}"
    log_lines.append(status)

Path("RUN_TEST_LOG.txt").write_text(status + "\n" + "\n".join(log_lines), encoding="utf-8")
print(status)
print("Quan sát: data/raw_images, data/processed_images, data/videos, outputs/image_metadata.csv, outputs/image_event_log.csv")

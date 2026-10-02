"""
ขั้นที่ 3 — เก็บภาพสำหรับทำ dataset
รันจากโฟลเดอร์หลักของโปรเจกต์:  python src/capture.py

ปุ่มควบคุม
  s = ถ่าย 1 ภาพ
  a = เปิด/ปิดโหมดถ่ายอัตโนมัติทุก 1 วินาที
  q = ออก

ภาพจะถูกเก็บไว้ที่ dataset/raw/
"""
import os
import time
from datetime import datetime

import cv2
from config import CAMERA_SOURCE, RAW_DIR

os.makedirs(RAW_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit("เชื่อมต่อกล้องไม่ได้ — ลองรัน src/test_cam.py ก่อน")

count = len([f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg")])
auto = False
last_auto = 0.0

print("s = ถ่าย | a = auto | q = ออก")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    now = time.time()
    save = False
    if auto and now - last_auto > 1.0:
        last_auto = now
        save = True

    view = frame.copy()
    color = (0, 255, 0) if auto else (0, 200, 255)
    cv2.putText(view, f"saved: {count}   auto: {'ON' if auto else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("capture", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        save = True
    elif key == ord("a"):
        auto = not auto
    elif key == ord("q"):
        break

    if save:
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        cv2.imwrite(os.path.join(RAW_DIR, name), frame)
        count += 1
        print(f"[{count}] {name}")

cap.release()
cv2.destroyAllWindows()
print(f"\nเก็บภาพทั้งหมด {count} ภาพ ไว้ที่ {RAW_DIR}")

"""
ขั้นที่ 2 — ทดสอบว่าต่อกล้องมือถือได้หรือยัง
รันจากโฟลเดอร์หลักของโปรเจกต์:  python src/test_cam.py
กด q เพื่อออก
"""
import cv2
from config import CAMERA_SOURCE

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit(
        "เชื่อมต่อกล้องไม่ได้\n"
        "- เช็คว่ามือถือกับคอมอยู่ Wi-Fi วงเดียวกัน\n"
        "- เช็คว่า IP ใน src/config.py ตรงกับที่แอปแสดง\n"
        "- ลองเปิด URL นั้นในเบราว์เซอร์ดูก่อน"
    )

print("ต่อกล้องสำเร็จ — กด q เพื่อออก")

while True:
    ok, frame = cap.read()
    if not ok:
        print("อ่านเฟรมไม่ได้ ลองใหม่...")
        continue

    h, w = frame.shape[:2]
    cv2.putText(frame, f"{w}x{h}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("camera test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

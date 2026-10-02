"""
แยกสี LED จากภาพ crop ด้วยค่า Hue เฉลี่ยใน HSV
ใช้ร่วมกันทั้งตอนสร้าง golden template และตอนตรวจสอบจริง

ใช้ circular mean (ค่าเฉลี่ยแบบมุมวงกลม) แทนการเฉลี่ยตรงๆ
เพราะ Hue ของสีแดงคาบเกี่ยวจุดเริ่มวงล้อสี (0 กับ 179 คือค่าใกล้กันจริงๆ)
ถ้าเฉลี่ยตรงๆ พิกเซลใกล้ 0 ปนกับใกล้ 179 จะได้ค่ากลางที่ผิดไปเป็นสีอื่น
"""
import cv2
import numpy as np

# มุม Hue อ้างอิงของแต่ละสี (สเกล OpenCV 0-179)
REFERENCE_HUES = {
    "red": 0,       # แดงอยู่ที่จุดเริ่มวงล้อ (0 ก็คือใกล้ 179 ด้วยเช่นกัน)
    "yellow": 25,
    "green": 60,
}


def _circular_distance(a, b, period=180):
    """ระยะห่างของมุมสองมุมบนวงกลม (คำนึงว่า 0 กับ 179 อยู่ใกล้กัน)"""
    d = abs(a - b) % period
    return min(d, period - d)


def classify_led_color(crop_bgr):
    """
    รับภาพ crop เฉพาะบริเวณ LED (BGR) คืนค่าเป็นชื่อสี: "red", "yellow", "green"
    หรือ "unknown" ถ้าไม่มีสีจริงๆ เลย (เช่น LED ดับสนิท เป็นสีเทาล้วน)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # ตัดพิกเซลที่จางเกินไป (แสงสะท้อนขาว/พื้นหลัง) ออก
    mask = (s > 40) & (v > 30)
    if mask.sum() < 5:
        mask = s > 15  # ผ่อนเกณฑ์ลงถ้าพิกเซลสีจัดน้อยเกินไป
        if mask.sum() < 5:
            return "unknown"

    # เฉลี่ยแบบ circular mean: แปลง Hue (0-179) เป็นมุมองศาเต็มวง (0-360) ก่อน
    hue_deg = h[mask].astype(float) * 2.0
    angles = np.deg2rad(hue_deg)
    mean_angle = np.degrees(
        np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    ) % 360
    mean_hue = mean_angle / 2.0  # กลับมาสเกล OpenCV 0-179

    # เทียบว่าใกล้สีอ้างอิงไหนที่สุดบนวงกลม
    best_color = min(
        REFERENCE_HUES,
        key=lambda c: _circular_distance(mean_hue, REFERENCE_HUES[c])
    )
    return best_color


def crop_from_bbox(image, bbox, shrink=0.2):
    """
    ตัดภาพเฉพาะบริเวณ bbox ออกมา โดยหดขอบเข้ามาเล็กน้อย (shrink)
    เพื่อลดโอกาสที่จะติดพื้นหลัง/ขาสาย LED ปนเข้ามาในการคำนวณสี
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = int(w * shrink / 2)
    dy = int(h * shrink / 2)

    x1s = x1 + dx
    y1s = y1 + dy
    x2s = x2 - dx
    y2s = y2 - dy

    if x2s <= x1s or y2s <= y1s:
        x1s, y1s, x2s, y2s = x1, y1, x2, y2

    return image[max(0, y1s):y2s, max(0, x1s):x2s]
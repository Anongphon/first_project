"""
เครื่องมือสร้าง Golden Template
วิธีใช้: ประกอบบอร์ดให้ถูกต้องสมบูรณ์ 100% ก่อน วางในกล้อง แล้วรันไฟล์นี้

หลักการ: detect อุปกรณ์บน "ภาพต้นฉบับ" (สัดส่วนเหมือนตอนเทรนโมเดล) เสมอ
แล้วค่อยแปลงพิกัดที่เจอไปเป็นพิกัดบนระบบ warp (800x300) ด้วยเมทริกซ์จาก board.py
เพื่อให้ยังเทียบตำแหน่งกันได้ตรงๆ โดยไม่ทำให้โมเดล detect ผิดพลาดจากภาพที่ถูกบีบสัดส่วน

ปุ่มควบคุม
  c = จับภาพปัจจุบันมาตรวจ แล้วแสดงผลให้ดูก่อนบันทึก
  s = บันทึกผลล่าสุดที่จับไว้ลง config/golden.json
  q = ออกโดยไม่บันทึก
"""
import json

import cv2

from config import (
    CAMERA_SOURCE, GOLDEN_PATH,
    CONF_THRESHOLD, WARP_WIDTH, WARP_HEIGHT,
)
from board import get_model, detect_and_warp, transform_points
from color_utils import classify_led_color, crop_from_bbox
from viz_utils import put_label


def detect_components(frame, matrix, model):
    """
    รัน YOLO บน "ภาพต้นฉบับ" (ไม่ใช่ภาพ warp) แล้วแปลงพิกัดที่เจอไปเป็นระบบ warp
    คืนรายการอุปกรณ์ (ไม่รวม breadboard เอง)
    แต่ละชิ้น: {class, bbox(พิกัด warp), center(พิกัด warp), conf, color(เฉพาะ led)}

    ใช้ชื่อคลาสจาก model.names โดยตรง ไม่เดาเลข index เอง
    เพื่อไม่ให้ผิดพลาดจากลำดับคลาสที่ Roboflow จัดไว้ตอนเทรน (มักเรียงตามตัวอักษร)
    """
    results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)[0]
    items = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name == "breadboard":
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # แปลง 4 มุมของ bbox ไปเป็นพิกัด warp แล้วหากรอบล้อมรอบใหม่
        corners_orig = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        warped_corners = transform_points(corners_orig, matrix)
        wx1, wy1 = warped_corners.min(axis=0)
        wx2, wy2 = warped_corners.max(axis=0)

        center_orig = [((x1 + x2) / 2, (y1 + y2) / 2)]
        wcx, wcy = transform_points(center_orig, matrix)[0]

        item = {
            "class": class_name,
            "bbox": [int(wx1), int(wy1), int(wx2), int(wy2)],
            "center": [int(wcx), int(wcy)],
            "conf": round(conf, 3),
        }

        if class_name == "led":
            # สีดูจากภาพต้นฉบับ คมชัดกว่าภาพ warp ที่ผ่านการ interpolate มา
            crop = crop_from_bbox(frame, (x1, y1, x2, y2))
            item["color"] = classify_led_color(crop)

        items.append(item)

    return items


def draw_preview(warped, items):
    vis = warped.copy()
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        label = item["class"]
        if "color" in item:
            label += f":{item['color']}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        put_label(vis, label, (x1, max(15, y1 - 6)))
    return vis


def summarize(items):
    counts = {}
    for item in items:
        key = item["class"]
        if "color" in item:
            key += f" ({item['color']})"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    print("กำลังโหลดโมเดล...")
    model = get_model()
    print("โหลดโมเดลสำเร็จ\n")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("เชื่อมต่อกล้องไม่ได้")

    print("วางบอร์ดที่ประกอบถูกต้องสมบูรณ์ในกล้อง")
    print("c = จับภาพตรวจดูก่อน | s = บันทึกเป็น golden | q = ออกโดยไม่บันทึก\n")

    last_items = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        warped, corners, conf, matrix = detect_and_warp(frame, model)

        if warped is not None:
            cv2.imshow("live (press c to capture)", warped)
        else:
            cv2.putText(frame, "No board detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("live (press c to capture)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("ออกโดยไม่บันทึก")
            break

        elif key == ord("c"):
            if warped is None:
                print("ยังไม่เจอบอร์ด ลองจัดตำแหน่งใหม่ก่อนกด c")
                continue

            last_items = detect_components(frame, matrix, model)
            preview = draw_preview(warped, last_items)
            cv2.imshow("preview (press s to save)", preview)

            print("=== พบอุปกรณ์ ===")
            for name, count in summarize(last_items).items():
                print(f"  {name}: {count}")
            print("ตรวจดูในหน้าต่างรูปว่าครบและถูกต้องไหม ก่อนกด s\n")

        elif key == ord("s"):
            if not last_items:
                print("ยังไม่ได้กด c เพื่อจับภาพตรวจก่อน")
                continue

            golden = {
                "board_size": [WARP_WIDTH, WARP_HEIGHT],
                "expected": last_items,
            }
            with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
                json.dump(golden, f, ensure_ascii=False, indent=2)

            print(f"บันทึก golden template ลง {GOLDEN_PATH} แล้ว")
            print(f"รวม {len(last_items)} ชิ้น")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
ขั้นที่ 8 — ตรรกะตรวจสอบ: รันตรวจจับบนภาพจริง แล้วเทียบกับ golden.json
ตรวจทั้งชนิด จำนวน ตำแหน่ง และสี (เฉพาะ LED)
"""
import json
import math

import cv2

from config import (
    GOLDEN_PATH, CONF_THRESHOLD, POSITION_TOLERANCE_PX,
)
from board import transform_points
from color_utils import classify_led_color, crop_from_bbox
from viz_utils import put_label


def load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_components_from_results(results, model, matrix, frame_for_color=None):
    """
    ดึงรายการอุปกรณ์ออกจากผลลัพธ์ YOLO ที่รันไว้แล้ว (ไม่รัน predict ซ้ำ)
    ใช้ตอนมี results จาก board.detect_and_warp(..., return_results=True) อยู่แล้ว
    เร็วกว่า detect_components() เกือบ 2 เท่า เพราะไม่ต้องรัน YOLO ซ้ำรอบสอง
    """
    items = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name == "breadboard":
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

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
            "orig_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "conf": round(conf, 3),
        }

        if class_name == "led":
            crop = crop_from_bbox(frame_for_color, (x1, y1, x2, y2)) if frame_for_color is not None else None
            item["color"] = classify_led_color(crop) if crop is not None else "unknown"

        items.append(item)

    return items


def detect_components(frame, matrix, model):
    """
    รัน YOLO บน "ภาพต้นฉบับ" (ไม่ใช่ภาพ warp) แล้วแปลงพิกัดที่เจอไปเป็นระบบ warp
    ด้วยเมทริกซ์เดียวกับที่ใช้สร้าง golden.json เพื่อให้เทียบกันได้ตรงๆ
    รูปแบบผลลัพธ์เดียวกับใน generate_golden.py

    ใช้ชื่อคลาสจาก model.names โดยตรง ไม่เดาเลข index เอง

    หมายเหตุ: ฟังก์ชันนี้รัน YOLO เอง 1 ครั้ง ถ้ามีผลลัพธ์จาก detect_and_warp
    อยู่แล้ว (return_results=True) ให้ใช้ detect_components_from_results() แทน
    จะเร็วกว่าเพราะไม่ต้องรันซ้ำ
    """
    results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)[0]
    return detect_components_from_results(results, model, matrix, frame_for_color=frame)


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compare_to_golden(detections, golden, tolerance=POSITION_TOLERANCE_PX):
    """
    เทียบผลตรวจจับกับ golden template

    หลักการจับคู่: สำหรับ golden แต่ละชิ้น หา detection ที่ตรงคลาส (และตรงสีถ้าเป็น led)
    ที่อยู่ใกล้ตำแหน่งเดิมที่สุดและอยู่ในระยะ tolerance ถ้าเจอ = ผ่าน ("ok")
    ถ้าไม่เจอเลยแต่มี detection คลาสเดียวกันอยู่ไกลเกิน tolerance = "wrong_position"
    ถ้าเป็น led ที่ตำแหน่งตรงแต่สีผิด = "wrong_color"
    ถ้าไม่เจอ detection ที่พอจะจับคู่ได้เลย = "missing"
    detection ที่เหลือหลังจับคู่ทั้งหมดแล้ว = "extra" (ของที่ไม่ควรอยู่ตรงนั้น)

    คืนค่า: dict มี overall_pass, matched, missing, wrong_position, wrong_color, extra
    """
    remaining = list(detections)  # จะตัดออกทีละตัวเมื่อจับคู่ได้แล้ว

    matched = []
    missing = []
    wrong_position = []
    wrong_color = []

    for g in golden["expected"]:
        g_class = g["class"]
        g_center = g["center"]
        g_color = g.get("color")

        # หา candidate ที่คลาสตรงกันทั้งหมดก่อน
        candidates = [d for d in remaining if d["class"] == g_class]

        if not candidates:
            missing.append(g)
            continue

        # ในกลุ่มคลาสเดียวกัน หาตัวที่ใกล้ตำแหน่ง golden ที่สุด
        candidates.sort(key=lambda d: _distance(d["center"], g_center))
        best = candidates[0]
        dist = _distance(best["center"], g_center)

        if g_class == "led":
            # LED ต้องเช็คสีด้วย ไม่ใช่แค่ตำแหน่ง
            same_color_candidates = [c for c in candidates if c.get("color") == g_color]
            if same_color_candidates:
                same_color_candidates.sort(key=lambda d: _distance(d["center"], g_center))
                best2 = same_color_candidates[0]
                dist2 = _distance(best2["center"], g_center)
                if dist2 <= tolerance:
                    matched.append({"golden": g, "detected": best2})
                    remaining.remove(best2)
                    continue
                else:
                    wrong_position.append({"golden": g, "detected": best2, "distance": round(dist2, 1)})
                    remaining.remove(best2)
                    continue
            else:
                # มี LED ตำแหน่งใกล้เคียงแต่สีผิด
                if dist <= tolerance:
                    wrong_color.append({"golden": g, "detected": best, "found_color": best.get("color")})
                    remaining.remove(best)
                    continue
                else:
                    missing.append(g)
                    continue

        # คลาสอื่นที่ไม่ใช่ led — เช็คแค่ตำแหน่ง
        if dist <= tolerance:
            matched.append({"golden": g, "detected": best})
            remaining.remove(best)
        else:
            wrong_position.append({"golden": g, "detected": best, "distance": round(dist, 1)})
            remaining.remove(best)

    extra = remaining  # เหลือจากจับคู่ = อุปกรณ์ที่ไม่ควรมี/ไม่ตรงกับ golden ตัวไหนเลย

    overall_pass = (
        len(missing) == 0
        and len(wrong_position) == 0
        and len(wrong_color) == 0
        and len(extra) == 0
    )

    return {
        "overall_pass": overall_pass,
        "matched": matched,
        "missing": missing,
        "wrong_position": wrong_position,
        "wrong_color": wrong_color,
        "extra": extra,
    }


def format_report(report):
    """แปลงผลเปรียบเทียบให้เป็นข้อความอ่านง่าย สำหรับพิมพ์ออก terminal"""
    lines = []
    status = "PASS ✓" if report["overall_pass"] else "FAIL ✗"
    lines.append(f"ผลตรวจสอบ: {status}")
    lines.append(f"  ตรงตามแบบ: {len(report['matched'])} ชิ้น")

    if report["missing"]:
        lines.append(f"  ขาดหาย: {len(report['missing'])} ชิ้น")
        for m in report["missing"]:
            label = m["class"]
            if "color" in m:
                label += f" สี{m['color']}"
            lines.append(f"    - ขาด {label} ที่ตำแหน่งประมาณ {m['center']}")

    if report["wrong_position"]:
        lines.append(f"  ผิดตำแหน่ง: {len(report['wrong_position'])} ชิ้น")
        for w in report["wrong_position"]:
            lines.append(
                f"    - {w['golden']['class']} ควรอยู่ {w['golden']['center']} "
                f"แต่เจอที่ {w['detected']['center']} (ห่าง {w['distance']}px)"
            )

    if report["wrong_color"]:
        lines.append(f"  สีผิด: {len(report['wrong_color'])} ชิ้น")
        for wc in report["wrong_color"]:
            lines.append(
                f"    - LED ตำแหน่ง {wc['golden']['center']} ควรเป็นสี{wc['golden']['color']} "
                f"แต่เจอสี{wc['found_color']}"
            )

    if report["extra"]:
        lines.append(f"  มีของเกิน/ไม่ควรอยู่ตรงนั้น: {len(report['extra'])} ชิ้น")
        for e in report["extra"]:
            label = e["class"]
            if "color" in e:
                label += f" สี{e['color']}"
            lines.append(f"    - เจอ {label} เกินที่ตำแหน่ง {e['center']}")

    return "\n".join(lines)


def draw_report(warped, report):
    """วาดผลตรวจสอบลงบนภาพ warp เพื่อแสดงผลตอนรันจริง"""
    vis = warped.copy()

    for m in report["matched"]:
        x1, y1, x2, y2 = m["detected"]["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for w in report["wrong_position"]:
        x1, y1, x2, y2 = w["detected"]["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        put_label(vis, "Wrong position", (x1, max(15, y1 - 6)), color=(0, 165, 255))

    for wc in report["wrong_color"]:
        x1, y1, x2, y2 = wc["detected"]["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        put_label(vis, f"Wrong color: {wc['found_color']}", (x1, max(15, y1 - 6)), color=(0, 165, 255))

    for e in report["extra"]:
        x1, y1, x2, y2 = e["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        put_label(vis, "Unexpected", (x1, max(15, y1 - 6)), color=(0, 0, 255))

    for m in report["missing"]:
        cx, cy = m["center"]
        cv2.drawMarker(vis, (cx, cy), (255, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        label = m["class"]
        if "color" in m:
            label += f":{m['color']}"
        put_label(vis, f"Missing {label}", (cx + 10, cy), color=(255, 0, 255))

    status_text = "PASS" if report["overall_pass"] else "FAIL"
    status_color = (0, 255, 0) if report["overall_pass"] else (0, 0, 255)
    put_label(vis, status_text, (10, 25), color=status_color, scale=0.9, thickness=2)

    return vis


def draw_report_on_frame(frame, report, inv_matrix):
    """
    เหมือน draw_report() แต่วาดผลตรวจสอบลงบน "ภาพกล้องดิบ" แทนภาพ warp
    ของที่ตรวจเจอ ใช้ orig_bbox (พิกัดจริงบนภาพกล้อง) วาดได้ตรงๆ
    ของที่ "ขาดหาย" ไม่มีพิกัดจริงอยู่แล้ว ต้องแปลงกลับจากพิกัดระบบ warp
    ด้วย inv_matrix (เมทริกซ์ผกผันของตัวที่ใช้ warp ภาพ) ก่อนจึงปักหมุดได้ถูกจุด
    """
    vis = frame.copy()

    for m in report["matched"]:
        x1, y1, x2, y2 = m["detected"]["orig_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for w in report["wrong_position"]:
        x1, y1, x2, y2 = w["detected"]["orig_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        put_label(vis, "Wrong position", (x1, max(15, y1 - 6)), color=(0, 165, 255))

    for wc in report["wrong_color"]:
        x1, y1, x2, y2 = wc["detected"]["orig_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        put_label(vis, f"Wrong color: {wc['found_color']}", (x1, max(15, y1 - 6)), color=(0, 165, 255))

    for e in report["extra"]:
        x1, y1, x2, y2 = e["orig_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        put_label(vis, "Unexpected", (x1, max(15, y1 - 6)), color=(0, 0, 255))

    for m in report["missing"]:
        orig_pt = transform_points([m["center"]], inv_matrix)[0]
        cx, cy = int(orig_pt[0]), int(orig_pt[1])
        cv2.drawMarker(vis, (cx, cy), (255, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
        label = m["class"]
        if "color" in m:
            label += f":{m['color']}"
        put_label(vis, f"Missing {label}", (cx + 10, cy), color=(255, 0, 255))

    status_text = "PASS" if report["overall_pass"] else "FAIL"
    status_color = (0, 255, 0) if report["overall_pass"] else (0, 0, 255)
    put_label(vis, status_text, (10, 30), color=status_color, scale=0.9, thickness=2)

    return vis


if __name__ == "__main__":
    # ทดสอบเดี่ยว: เปิดกล้อง หาบอร์ด ตรวจสอบ เทียบกับ golden แบบเรียลไทม์
    from config import CAMERA_SOURCE
    from board import get_model, detect_and_warp

    print("กำลังโหลดโมเดลและ golden template...")
    model = get_model()
    golden = load_golden()
    print(f"โหลดสำเร็จ golden มี {len(golden['expected'])} ชิ้น\n")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("เชื่อมต่อกล้องไม่ได้")

    print("กด q เพื่อออก")
    last_printed_status = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        warped, corners, conf, matrix = detect_and_warp(frame, model)

        if warped is not None:
            detections = detect_components(frame, matrix, model)
            report = compare_to_golden(detections, golden)
            vis = draw_report(warped, report)
            cv2.imshow("inspection result", vis)

            if report["overall_pass"] != last_printed_status:
                print(format_report(report))
                print("-" * 40)
                last_printed_status = report["overall_pass"]
        else:
            cv2.putText(frame, "No board detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("inspection result", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
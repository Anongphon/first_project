"""
ขั้นที่ 9 (เวอร์ชันปรับปรุง) — ระบบหลัก จำลองการทำงานแบบสายพานการผลิต

การปรับปรุงจากเวอร์ชันก่อน:
1. หน้าต่างแสดงผลขนาดคงที่เสมอ (ประกอบภาพลงบน canvas ขนาดตายตัว
   ไม่ว่าภาพต้นทางจะเป็นภาพกล้องดิบหรือภาพ warp ก็ไม่ทำให้หน้าต่างรีไซส์)
2. UI มีแถบหัวเรื่อง + กรอบภาพ + แถบสถานะพร้อม progress bar
3. เร็วขึ้น เพราะรัน YOLO แค่ 1 ครั้งต่อเฟรม (เดิมรัน 2 ครั้งซ้อนกัน)
   และลดขนาดภาพที่ป้อนเข้าโมเดล (ปรับได้ที่ config.INFER_IMGSZ)
"""
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, LOG_DIR,
    UI_WIDTH, UI_HEADER_HEIGHT, UI_CONTENT_HEIGHT, UI_FOOTER_HEIGHT,
    DETECT_EVERY_N_FRAMES,
)
from board import get_model, detect_and_warp, draw_board_outline, transform_points
from inspector import (
    load_golden, detect_components_from_results, compare_to_golden,
    draw_report_on_frame, format_report,
)
from viz_utils import put_label

# ---------- ค่าปรับแต่งจังหวะการทำงาน ----------
STABLE_FRAMES_REQUIRED = 8     # ต้องเจอบอร์ดต่อเนื่องกี่เฟรมถึงจะเริ่มตรวจ
RESULT_DISPLAY_SECONDS = 3.0   # โชว์ผล PASS/FAIL ค้างไว้กี่วินาที

STATE_WAITING = "WAITING"
STATE_RESULT = "RESULT"

WINDOW_NAME = "Board Inspector"

# สีธีม (BGR)
COLOR_BG = (32, 28, 24)
COLOR_HEADER = (60, 45, 30)
COLOR_FOOTER = (24, 22, 20)
COLOR_ACCENT = (255, 200, 0)
COLOR_PASS = (90, 200, 90)
COLOR_FAIL = (70, 70, 230)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (170, 170, 170)


def setup_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "inspection_log.csv")
    is_new = not os.path.exists(log_path)
    f = open(log_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["timestamp", "result", "missing", "wrong_position", "wrong_color", "extra"])
    return f, writer, log_path


def log_result(writer, file_handle, report):
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PASS" if report["overall_pass"] else "FAIL",
        len(report["missing"]),
        len(report["wrong_position"]),
        len(report["wrong_color"]),
        len(report["extra"]),
    ])
    file_handle.flush()


def fit_into_box(img, box_w, box_h, bg_color=COLOR_BG):
    """
    ย่อ/ขยายภาพให้พอดีกรอบขนาดคงที่ โดยรักษาสัดส่วนเดิม (letterbox)
    เหลือพื้นที่ว่างจะเติมสีพื้นหลังแทน ทำให้ผลลัพธ์มีขนาดคงที่เป๊ะเสมอ
    """
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full((box_h, box_w, 3), bg_color, dtype=np.uint8)
    x_off = (box_w - new_w) // 2
    y_off = (box_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_header():
    header = np.full((UI_HEADER_HEIGHT, UI_WIDTH, 3), COLOR_HEADER, dtype=np.uint8)
    put_label(header, "BOARD ASSEMBLY INSPECTOR", (20, 38),
              color=COLOR_TEXT, scale=0.8, thickness=2)
    now_text = datetime.now().strftime("%H:%M:%S")
    put_label(header, now_text, (UI_WIDTH - 110, 38),
              color=COLOR_TEXT_DIM, scale=0.7, thickness=1)
    return header


def draw_progress_bar(img, x, y, w, h, fraction, color):
    """แถบความคืบหน้า: กรอบขาวจาง + แท่งสีเติมตามสัดส่วน fraction (0.0-1.0)"""
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    fill_w = int(w * fraction)
    if fill_w > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill_w - 1, y + h - 1), color, -1)


def draw_footer_waiting(board_present, stable_count):
    footer = np.full((UI_FOOTER_HEIGHT, UI_WIDTH, 3), COLOR_FOOTER, dtype=np.uint8)

    if board_present:
        text = f"Item detected, confirming position... ({stable_count}/{STABLE_FRAMES_REQUIRED})"
        color = COLOR_ACCENT
        fraction = stable_count / STABLE_FRAMES_REQUIRED
    else:
        text = "Waiting for item..."
        color = COLOR_TEXT_DIM
        fraction = 0.0

    put_label(footer, text, (20, 35), color=color, scale=0.65, thickness=1)
    draw_progress_bar(footer, 20, 55, UI_WIDTH - 40, 18, fraction, COLOR_ACCENT)
    return footer


def _item_label(item):
    """แปลง item dict เป็นข้อความสั้นๆ เช่น 'led(red)' หรือ 'resistor'"""
    label = item["class"]
    if "color" in item:
        label += f"({item['color']})"
    return label


def build_detail_text(report):
    """
    สร้างข้อความบอกชัดๆ ว่าอะไรขาด/ผิดตำแหน่ง/สีผิด/เกิน แทนที่จะบอกแค่ตัวเลข
    เช่น "Missing: led(red), resistor  |  Wrong pos: button"
    """
    parts = []

    if report["missing"]:
        names = [_item_label(m) for m in report["missing"]]
        parts.append("Missing: " + ", ".join(names))

    if report["wrong_position"]:
        names = [w["golden"]["class"] for w in report["wrong_position"]]
        parts.append("Wrong pos: " + ", ".join(names))

    if report["wrong_color"]:
        names = [
            f"{wc['golden']['class']}(want {wc['golden']['color']} got {wc['found_color']})"
            for wc in report["wrong_color"]
        ]
        parts.append("Wrong color: " + ", ".join(names))

    if report["extra"]:
        names = [_item_label(e) for e in report["extra"]]
        parts.append("Extra: " + ", ".join(names))

    if not parts:
        return "All components correct"

    text = "   |   ".join(parts)
    max_chars = 110  # กันข้อความยาวเกินความกว้างหน้าต่างจนล้น
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    return text


def draw_footer_result(report, elapsed, total):
    footer = np.full((UI_FOOTER_HEIGHT, UI_WIDTH, 3), COLOR_FOOTER, dtype=np.uint8)

    status_pass = report["overall_pass"]
    badge_color = COLOR_PASS if status_pass else COLOR_FAIL
    badge_text = "PASS" if status_pass else "FAIL"

    # ป้าย PASS/FAIL สีพื้นตัน ด้านซ้าย
    badge_w = 130
    cv2.rectangle(footer, (20, 15), (20 + badge_w, 60), badge_color, -1)
    put_label(footer, badge_text, (20 + 18, 48), color=(255, 255, 255), scale=1.0, thickness=2)

    # สรุปจำนวนด้านขวาของป้าย
    summary_parts = [f"OK {len(report['matched'])}"]
    if report["missing"]:
        summary_parts.append(f"Missing {len(report['missing'])}")
    if report["wrong_position"]:
        summary_parts.append(f"Wrong pos {len(report['wrong_position'])}")
    if report["wrong_color"]:
        summary_parts.append(f"Wrong color {len(report['wrong_color'])}")
    if report["extra"]:
        summary_parts.append(f"Extra {len(report['extra'])}")
    summary_text = " | ".join(summary_parts)
    put_label(footer, summary_text, (20 + badge_w + 20, 42),
              color=COLOR_TEXT, scale=0.6, thickness=1)

    # บรรทัดรายละเอียด บอกชื่อชิ้นที่ผิดตรงๆ ไม่ใช่แค่ตัวเลข
    detail_text = build_detail_text(report)
    detail_color = COLOR_TEXT if status_pass else (0, 200, 255)
    put_label(footer, detail_text, (20, 78), color=detail_color, scale=0.55, thickness=1)

    # แถบนับถอยหลังเวลาที่จะแสดงผลค้างไว้
    remaining = max(0.0, total - elapsed)
    fraction = remaining / total
    draw_progress_bar(footer, 20, 110, UI_WIDTH - 40, 14, fraction, badge_color)
    put_label(footer, f"{remaining:.1f}s", (UI_WIDTH - 60, 106),
              color=COLOR_TEXT_DIM, scale=0.5, thickness=1)

    return footer


def compose_canvas(content_img, footer_img):
    header = draw_header()
    content_panel = fit_into_box(content_img, UI_WIDTH, UI_CONTENT_HEIGHT)
    return np.vstack([header, content_panel, footer_img])


def main():
    print("กำลังโหลดโมเดลและ golden template...")
    model = get_model()
    golden = load_golden()
    print(f"พร้อมทำงาน — golden มี {len(golden['expected'])} ชิ้น\n")

    log_file, log_writer, log_path = setup_log()
    print(f"บันทึกผลลง {log_path}\n")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("เชื่อมต่อกล้องไม่ได้")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    state = STATE_WAITING
    stable_count = 0
    result_shown_at = None
    last_report = None
    last_content_img = None

    # แคชผลตรวจจับล่าสุดไว้ใช้ระหว่างเฟรมที่ "ข้าม" การรัน YOLO (ตอน throttle)
    frame_counter = 0
    cached_board_present = False
    cached_corners = None
    cached_conf = None
    cached_matrix = None
    cached_results = None

    print("ระบบพร้อมทำงาน — วางชิ้นงานในกล้องเพื่อเริ่มตรวจสอบ")
    print("กด q เพื่อออก\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if state == STATE_RESULT:
            elapsed = time.time() - result_shown_at
            if elapsed < RESULT_DISPLAY_SECONDS:
                # ช่วงโชว์ผลค้างไว้ ภาพหยุดนิ่งอยู่แล้ว ไม่จำเป็นต้องรัน YOLO เลย
                # (ประหยัดงานไปได้เยอะมาก เพราะช่วงนี้กินเวลาหลายวินาทีทุกรอบ)
                footer = draw_footer_result(last_report, elapsed, RESULT_DISPLAY_SECONDS)
                canvas = compose_canvas(last_content_img, footer)
                cv2.imshow(WINDOW_NAME, canvas)
            else:
                # หมดเวลาโชว์ผลแล้ว รอให้เอาชิ้นงานออกก่อนถึงจะรับชิ้นถัดไป
                # เช็คแค่ทุกๆ N เฟรม ไม่ต้องเช็คถี่ทุกเฟรมก็ได้ ไม่กระทบการใช้งานจริง
                frame_counter += 1
                if frame_counter % DETECT_EVERY_N_FRAMES == 0:
                    warped, corners, conf, matrix, results = detect_and_warp(frame, model, return_results=True)
                    cached_board_present = warped is not None

                if not cached_board_present:
                    state = STATE_WAITING
                    stable_count = 0
                    print("รอชิ้นงานถัดไป...\n")
                else:
                    footer = np.full((UI_FOOTER_HEIGHT, UI_WIDTH, 3), COLOR_FOOTER, dtype=np.uint8)
                    put_label(footer, "Remove item to inspect next one", (20, 45),
                              color=(0, 165, 255), scale=0.65, thickness=1)
                    canvas = compose_canvas(last_content_img, footer)
                    cv2.imshow(WINDOW_NAME, canvas)

        if state == STATE_WAITING:
            frame_counter += 1
            run_detection_this_frame = (frame_counter % DETECT_EVERY_N_FRAMES == 0)

            if run_detection_this_frame:
                # รัน YOLO ครั้งเดียวต่อรอบ แล้วแชร์ผลลัพธ์ทั้งหาบอร์ดและหาอุปกรณ์
                warped, corners, conf, matrix, results = detect_and_warp(frame, model, return_results=True)
                cached_board_present = warped is not None
                cached_corners, cached_conf = corners, conf
                cached_matrix, cached_results = matrix, results
                stable_count = stable_count + 1 if cached_board_present else 0
            # เฟรมที่ไม่ได้รัน YOLO (throttle) ใช้ค่าที่แคชไว้ล่าสุดต่อไป
            # ภาพที่โชว์ยังเป็นเฟรมสดจากกล้อง แค่ไม่มีการตรวจจับใหม่ในเฟรมนั้นๆ

            if stable_count >= STABLE_FRAMES_REQUIRED:
                # ใช้ผลลัพธ์ YOLO ที่แคชไว้ล่าสุด ไม่ต้องรันซ้ำ
                detections = detect_components_from_results(
                    cached_results, model, cached_matrix, frame_for_color=frame
                )
                report = compare_to_golden(detections, golden)

                # แปลงตำแหน่ง "ของที่ขาด" กลับเป็นพิกัดบนภาพกล้องดิบ แล้ววาดผลทับบนภาพกล้องดิบเลย
                inv_matrix = np.linalg.inv(cached_matrix)
                vis = draw_report_on_frame(frame, report, inv_matrix)

                print(format_report(report))
                print("-" * 40)
                log_result(log_writer, log_file, report)

                last_report = report
                last_content_img = vis
                result_shown_at = time.time()
                state = STATE_RESULT

                footer = draw_footer_result(report, 0.0, RESULT_DISPLAY_SECONDS)
                canvas = compose_canvas(vis, footer)
                cv2.imshow(WINDOW_NAME, canvas)
            else:
                # โชว์ภาพกล้องดิบเสมอ (ไม่สลับไปภาพ warp) พร้อมกรอบเขียวถ้าเจอบอร์ด
                content_img = (
                    draw_board_outline(frame, cached_corners, cached_conf)
                    if cached_board_present else frame
                )
                footer = draw_footer_waiting(cached_board_present, stable_count)
                canvas = compose_canvas(content_img, footer)
                cv2.imshow(WINDOW_NAME, canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("ปิดโปรแกรมแล้ว")


if __name__ == "__main__":
    main()
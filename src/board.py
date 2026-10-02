
import cv2
import numpy as np
from ultralytics import YOLO

from config import MODEL_PATH, WARP_WIDTH, WARP_HEIGHT, CONF_THRESHOLD, INFER_IMGSZ
from viz_utils import put_label

_model = None


def get_model():
    """โหลดโมเดล YOLO ครั้งเดียวแล้วเก็บไว้ใช้ซ้ำ (โหลดใหม่ทุกเฟรมจะช้ามาก)"""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def order_corners(pts):
    """เรียงจุด 4 มุมให้อยู่ในลำดับ: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย"""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def run_inference(frame, model=None):
    """
    รัน YOLO บนภาพ 1 ครั้ง คืนผลลัพธ์ดิบ (ultralytics Results object)
    ให้โมดูลอื่นเอาไปใช้ต่อได้โดยไม่ต้องรันซ้ำ (ประหยัดเวลาไปเกือบครึ่ง
    เทียบกับเดิมที่รันแยกกันสำหรับ "หาบอร์ด" กับ "หาอุปกรณ์บนบอร์ด")
    """
    model = model or get_model()
    return model.predict(frame, verbose=False, conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ)[0]


def find_board_bbox_from_results(results, model):
    """
    ดึงกรอบ breadboard ออกจากผลลัพธ์ YOLO ที่รันไว้แล้ว (ไม่รัน predict ซ้ำ)
    คืนค่า: (x1, y1, x2, y2, conf) ของกล่องที่มั่นใจสุด หรือ None ถ้าไม่เจอ
    """
    best_box = None
    best_conf = 0.0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        if name == "breadboard" and conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return int(x1), int(y1), int(x2), int(y2), best_conf


def find_board_bbox(frame, model=None):
    """
    ใช้ YOLO หา breadboard ในภาพ (รัน inference เอง 1 ครั้ง)
    ใช้ตอนต้องการแค่หาบอร์ดเดี่ยวๆ ไม่ได้ต้องการผลอุปกรณ์อื่นด้วย
    """
    model = model or get_model()
    results = run_inference(frame, model)
    return find_board_bbox_from_results(results, model)


def refine_corners_in_bbox(frame, bbox, margin=20):
    """
    ภายในกรอบที่ YOLO บอก ลองหาขอบบอร์ดแม่นๆ ด้วย contour สีขาว
    ถ้าหาไม่ได้ ใช้มุมกรอบสี่เหลี่ยมของ YOLO ตรงๆ แทน (fallback)
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1e = max(0, x1 - margin)
    y1e = max(0, y1 - margin)
    x2e = min(w, x2 + margin)
    y2e = min(h, y2 + margin)

    crop = frame[y1e:y2e, x1e:x2e]
    fallback = order_corners(np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32"
    ))

    if crop.size == 0:
        return fallback

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 70, 255]))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback

    largest = max(contours, key=cv2.contourArea)
    crop_area = crop.shape[0] * crop.shape[1]
    if cv2.contourArea(largest) < crop_area * 0.3:
        return fallback

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(largest)
        pts = cv2.boxPoints(rect).astype("float32")

    pts[:, 0] += x1e
    pts[:, 1] += y1e
    return order_corners(pts)


def compute_perspective_matrix(corners):
    """คำนวณเมทริกซ์แปลงมุมมอง จาก 4 มุมบนภาพต้นฉบับ ไปเป็นภาพ warp ขนาดคงที่"""
    dst = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1],
    ], dtype="float32")
    return cv2.getPerspectiveTransform(corners, dst)


def transform_points(points, matrix):
    """
    แปลงจุด (x, y) จากพิกัดภาพต้นฉบับ ไปเป็นพิกัดบนภาพ warp
    points: list ของ (x, y)
    คืนค่า: numpy array รูปทรง (N, 2)
    """
    pts = np.array(points, dtype="float32").reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed.reshape(-1, 2)


def warp_board(frame, matrix):
    """ใช้เมทริกซ์ที่คำนวณไว้แปลงภาพทั้งภาพให้เป็นมุมมองบนลงล่าง"""
    return cv2.warpPerspective(frame, matrix, (WARP_WIDTH, WARP_HEIGHT))


def detect_and_warp(frame, model=None, return_results=False):
    """
    ฟังก์ชันรวม: YOLO หา breadboard -> หามุมแม่นๆ -> เตรียมเมทริกซ์ + ภาพ warp
    คืนค่า: (warped_image, corners, confidence, matrix)
    หรือ (None, None, None, None) ถ้าไม่เจอบอร์ด

    matrix เอาไว้แปลงพิกัดของอุปกรณ์ที่ detect จากภาพต้นฉบับ ไปเป็นพิกัดบน warp

    ถ้า return_results=True จะคืนผลลัพธ์ YOLO ดิบมาด้วย (ตัวที่ 5 ของ tuple)
    เพื่อให้ผู้เรียกเอาไปหาอุปกรณ์อื่นต่อได้ทันที โดยไม่ต้องรัน YOLO ซ้ำรอบสอง
    """
    model = model or get_model()
    results = run_inference(frame, model)
    bbox_info = find_board_bbox_from_results(results, model)

    if bbox_info is None:
        if return_results:
            return None, None, None, None, results
        return None, None, None, None

    x1, y1, x2, y2, conf = bbox_info
    corners = refine_corners_in_bbox(frame, (x1, y1, x2, y2))
    matrix = compute_perspective_matrix(corners)
    warped = warp_board(frame, matrix)

    if return_results:
        return warped, corners, conf, matrix, results
    return warped, corners, conf, matrix


def draw_board_outline(frame, corners, conf=None):
    """วาดกรอบบอร์ดที่เจอลงบนภาพต้นฉบับ (ใช้ debug/แสดงผลตอนรันจริง)"""
    if corners is None:
        return frame
    vis = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for p in pts:
        cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
    if conf is not None:
        x, y = pts[0]
        put_label(vis, f"breadboard {conf:.2f}", (int(x), max(20, int(y) - 8)),
                   color=(0, 255, 0), scale=0.8, thickness=2)
    return vis


if __name__ == "__main__":
    from config import CAMERA_SOURCE

    print("กำลังโหลดโมเดล...")
    model = get_model()
    print("โหลดโมเดลสำเร็จ")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("เชื่อมต่อกล้องไม่ได้")

    print("กด q เพื่อออก")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        warped, corners, conf, matrix = detect_and_warp(frame, model)
        outlined = draw_board_outline(frame, corners, conf)

        cv2.imshow("original + outline", outlined)
        if warped is not None:
            cv2.imshow("warped (top-down)", warped)
        else:
            cv2.putText(outlined, "ไม่เจอบอร์ด", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("original + outline", outlined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
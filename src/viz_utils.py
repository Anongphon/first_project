"""ฟังก์ชันช่วยวาดตัวอักษรบนภาพให้อ่านง่าย"""
import cv2


def put_label(img, text, org, color=(255, 0, 0), scale=0.5, thickness=2):
    """
    วาดข้อความสีน้ำเงิน (ค่าเริ่มต้น) ตัวหนาพอสมควร ให้อ่านง่ายบนพื้นบอร์ดสีขาว
    """
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)
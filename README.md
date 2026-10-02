# board-inspector

ระบบตรวจสอบการประกอบอุปกรณ์บน breadboard ด้วย Deep Learning + Computer Vision

## โครงสร้างโฟลเดอร์

```
board-inspector/
├── dataset/
│   ├── raw/            ภาพดิบที่ถ่ายมา (capture.py จะเซฟลงที่นี่)
│   └── yolo/           ภาพ + label ที่ดาวน์โหลดจาก Roboflow
├── models/             ไฟล์โมเดล best.pt หลังเทรนเสร็จ
├── config/
│   └── golden.json     แบบมาตรฐานของชิ้นงาน
├── logs/               บันทึกผลตรวจ
├── src/
│   ├── config.py       ตั้งค่ากลาง ← แก้ IP กล้องที่นี่
│   ├── test_cam.py     ทดสอบต่อกล้อง
│   ├── capture.py      เก็บภาพทำ dataset
│   ├── board.py        หาขอบบอร์ด + warp
│   ├── inspector.py    ตรรกะตรวจสอบ
│   └── main.py         ระบบหลัก
├── requirements.txt
└── README.md
```

## วิธีเริ่มใช้งาน

เปิด terminal แล้ว cd เข้ามาในโฟลเดอร์นี้ก่อน จากนั้น:

**1. สร้าง virtual environment**

Windows
```
python -m venv venv
venv\Scripts\activate
```

macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```

**2. ติดตั้งไลบรารี**
```
pip install -r requirements.txt
```

**3. แก้ IP ของกล้อง**

เปิด `src/config.py` แล้วแก้บรรทัด `CAMERA_SOURCE` ให้ตรงกับ URL ที่แอป IP Webcam
แสดงบนหน้าจอมือถือ อย่าลืมเติม `/video` ต่อท้าย

**4. ทดสอบกล้อง**
```
python src/test_cam.py
```

**5. เริ่มเก็บภาพ**
```
python src/capture.py
```

> ทุกคำสั่งต้องรันจากโฟลเดอร์ `board-inspector` นี้ ไม่ใช่จากในโฟลเดอร์ `src`

from picamera2 import Picamera2
from ultralytics import YOLO
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import time
import os
from pathlib import Path

# 1. 모델 경로 설정
MODEL_PATH = "/home/pi/A_TEST_CAN/best.pt"

# 2. YOLO 모델 로드
model = YOLO(MODEL_PATH)
print("모델 로드 완료:", model.names)

# 3. 카메라 시작
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

time.sleep(1)

try:
    while True:
        # 카메라 프레임 읽기
        frame = picam2.capture_array()

        # YOLO 감지
        results = model(frame, conf=0.35, verbose=False)

        # 감지 결과 네모 박스 그리기
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 네모 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 라벨 텍스트
                label = f"{name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Jupyter 화면 출력
        clear_output(wait=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame)
        plt.axis("off")
        plt.show()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("감지 종료")

finally:
    picam2.stop()

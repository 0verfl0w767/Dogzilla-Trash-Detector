from picamera2 import Picamera2
from ultralytics import YOLO
from xgolib import XGO
import time

MODEL_PATH = "/home/pi/A_TEST_CAN/old_best.pt"

CAN_CLASSES = {"CAN", "Can", "Cans", "can", "METAL", "Metal"}

CONF_THRES = 0.25
IMG_SIZE = 320
DETECT_EVERY = 2          # 2프레임마다 탐지
REQUIRED_HITS = 2         # 연속 2번 감지되면 작동
COOLDOWN_SEC = 8          # 집게 반복 작동 방지

# 몸 / 팔 위치 튜닝값
BODY_Z_NORMAL = 100       # 기본 몸 높이
BODY_Z_LOW = 45           # 몸 최대한 낮추기, 너무 낮으면 80으로 올려보기

ARM_READY_X = 100
ARM_READY_Z = -30

ARM_GRAB_X = 155          # 더 안 닿으면 155까지 가능
ARM_GRAB_Z = -95          # 더 안 닿으면 -95까지 가능

ARM_LIFT_X = 80
ARM_LIFT_Z = 80

# Dogzilla Lite 제어 객체
dog = XGO(port="/dev/ttyAMA0", version="xgolite")

model = YOLO(MODEL_PATH)
print("모델 로드 완료:", model.names)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (256, 192)}
)
picam2.configure(config)
picam2.start()

time.sleep(1)

def grab_can():
    print("캔 감지됨 -> 집게 동작 시작")

    try:
        # 팔 안정화 모드 끄기
        # 이걸 꺼야 몸을 낮출 때 팔/집게도 같이 내려감
        dog.arm_mode(0)
        time.sleep(0.3)

        # 몸 낮추기: 다리 오그라드는 효과
        dog.translation('z', BODY_Z_LOW)
        time.sleep(0.8)

        # 앞쪽 숙이기
        dog.attitude('p', 15)
        time.sleep(0.5)

        # 집게 최대 열림
        # 0이 최대 열림, 255가 최대 닫힘
        dog.claw(0)
        time.sleep(0.7)

        # 팔을 캔 쪽으로 먼저 접근
        dog.arm(ARM_READY_X, ARM_READY_Z)
        time.sleep(0.8)

        # 팔을 더 앞으로 + 아래로 내려서 캔 잡는 위치
        dog.arm(ARM_GRAB_X, ARM_GRAB_Z)
        time.sleep(1.0)

        # 집게 닫기
        dog.claw(255)
        time.sleep(0.8)

        # 캔 들어올리기
        dog.arm(ARM_LIFT_X, ARM_LIFT_Z)
        time.sleep(1.0)

        dog.attitude('p', 0)
        time.sleep(0.5)

        # 몸 다시 기본 높이로
        dog.translation('z', BODY_Z_NORMAL)
        time.sleep(0.8)

        print("집게 동작 완료")

    except Exception as e:
        print("집게 동작 중 오류:", e)

frame_count = 0
detect_hits = 0
last_grab_time = 0

try:
    print("캔 탐지 시작")

    while True:
        frame = picam2.capture_array()
        frame_count += 1

        # 매 프레임 YOLO 돌리지 않고 2프레임마다 탐지
        if frame_count % DETECT_EVERY != 0:
            time.sleep(0.03)
            continue

        results = model(
            frame,
            conf=CONF_THRES,
            imgsz=IMG_SIZE,
            max_det=5,
            verbose=False
        )

        can_detected = False
        best_name = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])

                if name not in CAN_CLASSES:
                    continue

                can_detected = True

                if conf > best_conf:
                    best_conf = conf
                    best_name = name

        if can_detected:
            detect_hits += 1
            print(f"감지 {detect_hits}/{REQUIRED_HITS}: {best_name} {best_conf:.2f}")
        else:
            detect_hits = 0

        now = time.time()

        if detect_hits >= REQUIRED_HITS and (now - last_grab_time) > COOLDOWN_SEC:
            grab_can()
            last_grab_time = now
            detect_hits = 0

        time.sleep(0.03)

except KeyboardInterrupt:
    print("감지 종료")

finally:
    picam2.stop()
    picam2.close()
    print("카메라 종료 완료")

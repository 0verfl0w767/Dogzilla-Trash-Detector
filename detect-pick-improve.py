from picamera2 import Picamera2
from ultralytics import YOLO
from xgolib import XGO
import time

# =========================
# NCNN 모델 경로
# =========================
MODEL_PATH = "/home/pi/A_TEST_CAN/best_ncnn_model"

# 네 모델 클래스:
# 0: glass
# 1: metal
# 2: paper
# 3: plastic
TARGET_CLASSES = {"glass", "metal", "paper", "plastic"}

CONF_THRES = 0.30
IMG_SIZE = 416
DETECT_EVERY = 2          # 2프레임마다 탐지
REQUIRED_HITS = 2         # 연속 2번 감지되면 작동
COOLDOWN_SEC = 8          # 집게 반복 작동 방지
MAX_DET = 5

# =========================
# 몸 / 팔 위치 튜닝값
# =========================
BODY_Z_NORMAL = 100

# 공식 범위상 z는 75 근처가 안전함
# 네 기체에서 45가 실제로 잘 먹으면 45로 바꿔도 됨
BODY_Z_LOW = 75

ARM_READY_X = 100
ARM_READY_Z = -30

ARM_GRAB_X = 155
ARM_GRAB_Z = -95

ARM_LIFT_X = 80
ARM_LIFT_Z = 80

# 집게
CLAW_OPEN = 0
CLAW_CLOSE = 255

# Dogzilla Lite 제어 객체
dog = XGO(port="/dev/ttyAMA0", version="xgolite")

# NCNN 모델 로드
model = YOLO(MODEL_PATH)
print("NCNN 모델 로드 완료:", model.names)

# 카메라 시작
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}
)
picam2.configure(config)
picam2.start()

time.sleep(1)

def grab_object(target_name="object"):
    print(f"{target_name} 감지됨 -> 집게 동작 시작")

    try:
        # 이동 중이면 정지
        dog.stop()
        time.sleep(0.2)

        # 팔 안정화 모드 끄기
        # 몸을 낮출 때 팔/집게도 같이 내려가게 함
        dog.arm_mode(0)
        time.sleep(0.3)

        # 몸 낮추기
        dog.translation('z', BODY_Z_LOW)
        time.sleep(0.8)

        # 앞쪽 숙이기
        # 반대로 올라가면 15를 -15로 바꿔보기
        dog.attitude('p', 15)
        time.sleep(0.5)

        # 집게 최대 열림
        dog.claw(CLAW_OPEN)
        time.sleep(0.7)

        # 팔을 대상 쪽으로 먼저 접근
        dog.arm(ARM_READY_X, ARM_READY_Z)
        time.sleep(0.8)

        # 팔을 더 앞으로 + 아래로 내려서 잡는 위치
        dog.arm(ARM_GRAB_X, ARM_GRAB_Z)
        time.sleep(1.0)

        # 집게 닫기
        dog.claw(CLAW_CLOSE)
        time.sleep(0.8)

        # 들어올리기
        dog.arm(ARM_LIFT_X, ARM_LIFT_Z)
        time.sleep(1.0)

        # 몸 기울기 복구
        dog.attitude('p', 0)
        time.sleep(0.5)

        # 몸 다시 기본 높이로
        dog.translation('z', BODY_Z_NORMAL)
        time.sleep(0.8)

        print("집게 동작 완료")

    except Exception as e:
        print("집게 동작 중 오류:", e)

def detect_target(frame):
    results = model(
        frame,
        conf=CONF_THRES,
        imgsz=IMG_SIZE,
        max_det=MAX_DET,
        verbose=False
    )

    best_target = None
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = str(model.names[cls_id]).strip().lower()
            conf = float(box.conf[0])

            if name not in TARGET_CLASSES:
                continue

            if conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_conf = conf
                best_target = {
                    "name": name,
                    "conf": conf,
                    "box": (x1, y1, x2, y2)
                }

    return best_target

frame_count = 0
detect_hits = 0
last_grab_time = 0
last_target = None

try:
    print("NCNN 재활용 쓰레기 탐지 시작")
    print("대상 클래스:", TARGET_CLASSES)

    while True:
        frame = picam2.capture_array()
        frame_count += 1

        # 매 프레임 YOLO 돌리지 않고 2프레임마다 탐지
        if frame_count % DETECT_EVERY != 0:
            time.sleep(0.03)
            continue

        start_time = time.time()
        target = detect_target(frame)
        infer_time = time.time() - start_time

        if target is not None:
            detect_hits += 1
            last_target = target

            print(
                f"감지 {detect_hits}/{REQUIRED_HITS}: "
                f"{target['name']} "
                f"{target['conf']:.2f} "
                f"box={target['box']} "
                f"infer={infer_time:.3f}s"
            )

        else:
            detect_hits = 0
            last_target = None

        now = time.time()

        if detect_hits >= REQUIRED_HITS and (now - last_grab_time) > COOLDOWN_SEC:
            if last_target is not None:
                grab_object(last_target["name"])

            last_grab_time = now
            detect_hits = 0
            last_target = None

        time.sleep(0.03)

except KeyboardInterrupt:
    print("탐지 종료")

finally:
    try:
        dog.stop()
        dog.attitude('p', 0)
        dog.translation('z', BODY_Z_NORMAL)
        dog.arm(ARM_LIFT_X, ARM_LIFT_Z)
        dog.claw(CLAW_OPEN)
    except:
        pass

    picam2.stop()
    picam2.close()
    print("카메라 종료 완료")

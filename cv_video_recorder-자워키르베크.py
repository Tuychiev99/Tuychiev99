import cv2
import argparse
import time

# 커맨드 라인 인자 설정
parser = argparse.ArgumentParser(description='Video Recorder')
parser.add_argument('--source', type=str, default=0, help='Camera source (0 for default camera)')
parser.add_argument('--fps', type=int, default=30, help='Frames per second')
parser.add_argument('--codec', type=str, default='MJPG', help='Video codec (MJPG, XVID, etc)')
parser.add_argument('--brightness', type=int, default=0, help='Brightness adjustment (-100 to 100)')
parser.add_argument('--contrast', type=int, default=0, help='Contrast adjustment (-100 to 100)')
parser.add_argument('--flip', type=int, default=0, help='Flip mode (0: none, 1: horizontal, 2: vertical)')
args = parser.parse_args()

# 카메라 초기화
cap = cv2.VideoCapture(args.source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 녹화 설정
fourcc = cv2.VideoWriter_fourcc(*args.codec)
out = None
is_recording = False
recording_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 필터 적용
    frame = cv2.convertScaleAbs(frame, alpha=1 + args.contrast/100, beta=args.brightness)
    if args.flip in [1, 2]:
        frame = cv2.flip(frame, args.flip-1)

    # 녹화 상태 표시
    if is_recording:
        cv2.circle(frame, (50,50), 20, (0,0,255), -1)
        recording_time = time.time() - recording_start_time
        cv2.putText(frame, f"REC {int(recording_time)}s", (80,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 화면 출력
    cv2.imshow('Video Recorder', frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break
    elif key == 32:  # Space 키
        if not is_recording:
            # 녹화 시작
            out = cv2.VideoWriter(
                f'recording_{time.strftime("%Y%m%d_%H%M%S")}.avi',
                fourcc,
                args.fps,
                (int(cap.get(3)), int(cap.get(4)))
            )
            recording_start_time = time.time()
            is_recording = True
        else:
            # 녹화 중지
            out.release()
            out = None
            is_recording = False

    # 프레임 저장
    if is_recording and out is not None:
        out.write(frame)

# 종료 처리
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

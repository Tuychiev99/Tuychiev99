import numpy as np
import cv2
import glob
import time # 동영상 처리 시간 조절을 위해 추가

# --- 설정 (Configuration) ---
# 체스보드의 내부 코너 개수 (가로 x 세로)
# 예: 10x7 사각형 체스보드는 내부 코너가 9x6개 입니다. 사용한 체스보드에 맞게 수정하세요.
CHESS_BOARD_WIDTH = 9
CHESS_BOARD_HEIGHT = 6
# 체스보드 사각형의 실제 크기 (mm 단위 권장, 임의 단위도 가능하나 일관성 유지)
SQUARE_SIZE = 25 # 예: 25mm

# 동영상 파일 경로
VIDEO_FILE = '/Users/tuychievjavokhirbek/Downloads/microscope-eyepiece-reticle-ne15-chessboard.jpg'

# 캘리브레이션 이미지 간격 (프레임 단위, 너무 촘촘하면 비슷하므로 적당히 건너뛰기)
FRAME_INTERVAL = 15
# --------------------------

# 체스보드 코너 개수 튜플
chessboard_size = (CHESS_BOARD_WIDTH, CHESS_BOARD_HEIGHT)

# 3D 월드 좌표계에서 체스보드 코너 위치 준비 (z=0 평면)
# (0,0,0), (1,0,0), ..., (width-1, height-1, 0) 형태의 배열 생성
objp = np.zeros((CHESS_BOARD_WIDTH * CHESS_BOARD_HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_BOARD_WIDTH, 0:CHESS_BOARD_HEIGHT].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE # 실제 크기 적용


objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# 동영상 파일 열기
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"Error: Cannot open video file: {VIDEO_FILE}")
    exit()

print("Starting calibration process...")
frame_count = 0
found_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    frame_count += 1
    # FRAME_INTERVAL 마다 프레임 처리
    if frame_count % FRAME_INTERVAL != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    # ret_corners: 코너 검출 성공 여부 (boolean)
    # corners: 검출된 코너의 픽셀 좌표 (numpy array)
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 코너를 찾았다면
    if ret_corners == True:
        found_count += 1
        objpoints.append(objp)

        # 코너 위치 정확도 향상 (서브픽셀 코너)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (선택 사항) 찾은 코너 시각화
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
        cv2.imshow('Chessboard Corners', frame)
        # cv2.waitKey(1) # 실시간 확인 시 주석 해제, 처리 속도 느려짐
        print(f"Frame {frame_count}: Chessboard found! ({found_count} successful detections)")
    else:
        # cv2.imshow('Chessboard Corners', frame) # 코너 못찾는 프레임 확인 시 주석 해제
        # cv2.waitKey(1)
        print(f"Frame {frame_count}: Chessboard not found.")
        pass # 코너 못찾으면 넘어감

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 처리 완료 후
cap.release()
cv2.destroyAllWindows()

if not imgpoints:
    print("Error: No valid chessboard corners found in the video. Calibration failed.")
    print("Try using a clearer video, better lighting, or adjust FRAME_INTERVAL.")
    exit()

if len(imgpoints) < 10: # 충분한 데이터를 확보했는지 확인 (최소 10개 이상 권장)
     print(f"Warning: Only {len(imgpoints)} views found. Calibration results might be inaccurate. More views are recommended.")

print(f"\nFound {len(imgpoints)} usable views for calibration.")
print("Performing camera calibration...")

# 카메라 캘리브레이션 수행
# ret: 성공 여부
# mtx: 카메라 매트릭스 (fx, fy, cx, cy)
# dist: 왜곡 계수 (k1, k2, p1, p2, k3)
# rvecs: 각 이미지의 회전 벡터 (Rotation vectors)
# tvecs: 각 이미지의 이동 벡터 (Translation vectors)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("\nCalibration successful!")
    print("\nCamera Matrix (mtx):\n", mtx)
    print(f"fx = {mtx[0, 0]:.4f}")
    print(f"fy = {mtx[1, 1]:.4f}")
    print(f"cx = {mtx[0, 2]:.4f}")
    print(f"cy = {mtx[1, 2]:.4f}")

    print("\nDistortion Coefficients (dist):\n", dist)
    # k1, k2, p1, p2, k3
    print(f"k1 = {dist[0, 0]:.4f}")
    print(f"k2 = {dist[0, 1]:.4f}")
    print(f"p1 = {dist[0, 2]:.4f}")
    print(f"p2 = {dist[0, 3]:.4f}")
    print(f"k3 = {dist[0, 4]:.4f}")


    # 재투영 오차 (Reprojection Error - RMSE) 계산
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    rmse = np.sqrt(mean_error / len(objpoints))
    print(f"\nTotal Mean Error (RMSE - Root Mean Square Error): {rmse:.4f} pixels")

   
else:
    print("\nCalibration failed.")
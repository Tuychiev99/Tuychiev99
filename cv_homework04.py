import cv2
import numpy as np

# 카메라 매트릭스와 왜곡 계수 로드
camera_matrix = np.load('/Users/tuychievjavokhirbek/Downloads/microscope-eyepiece-reticle-ne15-chessboard.jpg')
dist_coeffs = np.load('dist_coeffs.npy')

# AR 물체의 3D 포인트 정의
object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 체스보드 코너 찾기 (예시)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        # 3D-2D 매핑
        _, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

        # AR 물체 렌더링 (예: 사각형)
        axis = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        frame = cv2.polylines(frame, [np.int32(imgpts)], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('AR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

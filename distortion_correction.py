import cv2
import numpy as np


mtx = np.array([[1000.0,    0.0, 960.0],
                [   0.0, 1000.0, 540.0],
                [   0.0,    0.0,   1.0]])

# dist = np.array([[k1, k2, p1, p2, k3]])
dist = np.array([[0.1, -0.05, 0.001, 0.001, 0.01]])


INPUT_FILE = 'data/chessboard.avi' 
OUTPUT_FILE = 'output_undistorted.avi' 

is_video = INPUT_FILE.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))
# --------------------------

if is_video:
    # 동영상 처리
    cap = cv2.VideoCapture(INPUT_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {INPUT_FILE}")
        exit()

    # 원본 동영상 정보 얻기 (프레임 크기, FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video info: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # 최적의 새 카메라 매트릭스 계산 및 ROI(관심 영역) 얻기
    # alpha=1: 왜곡 보정 시 검은 영역이 생기더라도 모든 픽셀 유지
    # alpha=0: 왜곡 보정 후 이미지 잘라내어 검은 영역 최소화
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
    print("\nOptimal New Camera Matrix:\n", newcameramtx)
    print("Region of Interest (ROI) after undistortion:", roi)
    x, y, w, h = roi

    # 결과 동영상 저장을 위한 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 코덱 설정 (다른 코덱 사용 가능: 'MJPG', 'MP4V' 등)
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height)) # 원본 크기로 저장

    print(f"\nProcessing video and saving undistorted result to {OUTPUT_FILE}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 왜곡 보정 수행 (빠른 방법)
        # dst_fast = cv2.undistort(frame, mtx, dist, None, mtx) # 원본 mtx 사용

        # 2. 왜곡 보정 수행 (최적화된 방법 + ROI 사용)
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        
        out.write(dst)

        
        frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        dst_small = cv2.resize(dst, (0,0), fx=0.5, fy=0.5)
        combined_display = np.hstack((frame_small, dst_small))
        cv2.imshow('Original vs Undistorted (Resized)', combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'): # 처리 중 'q' 누르면 중지
            break

    # 작업 완료 후 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

else:
    # 이미지 처리
    img = cv2.imread(INPUT_FILE)
    if img is None:
        print(f"Error: Cannot read image file: {INPUT_FILE}")
        exit()

    h, w = img.shape[:2]
    print(f"Input image info: {w}x{h}")

    # 최적의 새 카메라 매트릭스 계산 및 ROI 얻기
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("\nOptimal New Camera Matrix:\n", newcameramtx)
    print("Region of Interest (ROI) after undistortion:", roi)
    x, y, w_roi, h_roi = roi

    # 왜곡 보정 수행
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

   
    # ROI 자르지 않고 전체 이미지 저장
    cv2.imwrite(OUTPUT_FILE, dst)
    print(f"\nUndistorted image saved to {OUTPUT_FILE}")

    
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', dst)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
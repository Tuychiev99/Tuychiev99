# Video Recorder 프로젝트

## 주요 기능
### 필수 기능
- 실시간 웹캠 프리뷰
- Space 키로 녹화 시작/정지
- ESC 키로 프로그램 종료
- 화면에 녹화 상태 표시 (빨간 원 및 녹화 시간)

### 추가 기능
- **화질 조정**
  - FPS 설정 (--fps)
  - 코덱 선택 (--codec)
- **실시간 필터**
  - 밝기/대비 조정 (--brightness, --contrast)
  - 화면 반전 (--flip)
- IP 카메라 지원 (RTSP 주소 입력)

## 사용 방법
```bash
# 기본 카메라 사용
python video_recorder.py

# IP 카메라 연결 (예: 천안시 교통 CCTV)
python video_recorder.py --source "rtsp://example.com/stream"

# 고급 설정 예시
python video_recorder.py --fps 60 --codec XVID --brightness 20 --flip 1
```

## 단축키
- Space: 녹화 시작/중지
- ESC: 프로그램 종료

## 의존성 설치
```bash
pip install opencv-python
```

## 지원 코덱 목록
- MJPG (기본값)
- XVID
- DIVX
- MP4V
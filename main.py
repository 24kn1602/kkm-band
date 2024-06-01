'''from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2'''
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

model_path = 'gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
base_options = BaseOptions(model_asset_path=model_path)
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 MediaPipe 이미지 객체로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # 현재 시간(ms)을 가져옴
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # 제스처 인식 수행
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # 결과 출력 및 프레임 표시
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
'''
# Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(0)
# Create a loop to read the latest frame from the camera using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    
    # Send live image data to perform gesture recognition.
# The results are accessible via the `result_callback` provided in
# the `GestureRecognizerOptions` object.
# The gesture recognizer must be created with the live stream mode.
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
'''
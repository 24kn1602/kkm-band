import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import playsound

# 모델 경로
model_path = 'gesture_recognizer.task'




# 옵션 설정
BaseOptions = mp.tasks.BaseOptions
base_options = BaseOptions(model_asset_path=model_path)
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


canplay = True

# 제스처 인식 결과를 출력하는 함수
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #print('gesture recognition result: {}'.format(result))
    #gesturename = result["GestureRecognizerResult"]["Gestures"]["Categories"][0]["categoryName"] == 'Open_Palm'
    global canplay

    print(canplay)
    if result.gestures != []:
        gesturename = result.gestures[0][0].category_name
        if bool(canplay):
            if gesturename == 'Thumb_Up':
                playsound.playsound('FX_piano01.mp3', block=False)
                canplay = False
            elif gesturename == 'Closed_Fist':
                playsound.playsound('FX_piano03.mp3', block=False)
                canplay = False
            elif gesturename == 'Open_Palm':
                playsound.playsound('FX_piano05.mp3', block=False)
                canplay = False
            elif gesturename == 'Pointing_Up':
                playsound.playsound('FX_piano06.mp3', block=False)
                canplay = False
            elif gesturename == 'Thumb_Down':
                playsound.playsound('FX_piano08.mp3', block=False)
                canplay = False
            elif gesturename == 'Victory':
                playsound.playsound('FX_piano10.mp3', block=False)
                canplay = False
            else:
                canplay = True 
        elif gesturename == 'None':
            canplay = True
    else:
        canplay = True


    
# 제스처 인식 옵션 설정
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands = 2
)

# 제스처 인식기 생성
with GestureRecognizer.create_from_options(options) as recognizer:
    # OpenCV를 사용하여 웹캠 캡처 시작
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
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # 결과 출력 및 프레임 표시
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

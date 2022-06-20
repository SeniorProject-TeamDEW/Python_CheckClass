from tabnanny import verbose
import cv2
import tensorflow.keras
import numpy as np
import paho.mqtt.client as mqtt
import json

def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)

    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위한 reshape
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

client = mqtt.Client()
client.connect("172.20.10.14", 1883, 60)

#client.on_connect = on_connect
#client.on_message = on_message

evt = json.loads('{"d":{}}')

# 동일한 class값이 연속해서 보내지는 문제 해결을 위한 알고리즘 설계
## old_class값과 new_class값이 같지 않을때만 class를 publish 하도록 설계.
old_class = str(0)

## 학습된 모델 불러오기
model_filename = '/Users/wonaz/Desktop/SeniorProject/converted_keras/keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0= 내장 카메라, 1 =외장 카메라 안될시 2로 사용
## 맥북에 카메라 연결시 내장 카메라로 바로 인식 가능
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 448)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)


while True:
    ret, frame = capture.read()

    '''
    if ret == True: 
        print("Read success!")
    '''

    # 이미지 뒤집기
    frame_fliped = cv2.flip(frame, 1)
    
    # 이미지 출력
    cv2.imshow("VideoFrame", frame_fliped)
    
    # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
    if cv2.waitKey(200) > 0: 
        break
    
    # 데이터 전처리
    preprocessed = preprocessing(frame_fliped)

    # 예측
    prediction = model.predict(preprocessed, verbose = False)
    #print(prediction) # [[0.00533728 0.99466264]]


    index = np.argmax(prediction)
    if prediction[0, index] >= 0.8:
        new_class = str(index)
        print(index)

    #client.loop()
    if old_class != new_class:
        evt['d']['class'] = new_class
        client.publish("iot-2/type/Python/id/Class/evt/status/fmt/json", json.dumps(evt))
        old_class = new_class # old_class refresh
   
    
# 카메라 객체 반환
capture.release() 
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()
import cv2 as cv
import tensorflow as tf
import nep
import sys
import threading
import numpy as np
import time 

def thread_function(name):
    global img, sub_image, pub_image
    while True:
        # Read data in a non-blocking mode
        s, msg = sub_image.listen() 
        # if s == True, then there is data in the socket      
        if s:
            img = msg
        else:
            time.sleep(.001)

node = nep.node('pose_points_listener')
conf = node.hybrid("127.0.0.1") # change the IP displayed when using nep master wifi
sub_image = node.new_sub('75_points','json',conf)
x = threading.Thread(target=thread_function, args=(1,))
x.start()
time.sleep(3)

model_path = r"C:\codes\class\presotsuron\presentation\model\model\training_relu"
model = tf.keras.models.load_model(model_path)
landmarks = np.zeros(75)
detect_history=np.zeros(11)
classes = np.zeros(10) #最大クラス数
while True:
    for i in range(75):
        landmarks[i] = img[str(i)]
    landmarks_processed = landmarks.reshape(1,75)
    predict_result = model.predict(landmarks_processed)
    print(np.squeeze(predict_result))
    predict_result = np.argmax(np.squeeze(predict_result))
    print("predicted: ",np.argmax(np.squeeze(predict_result)))
    for i in range(1,11):
        detect_history[11-i] = detect_history[10-i]
    for i in range(11):
        classes[int(detect_history[i])] += 1
    result = np.argmax(classes)
    print(result)
    time.sleep(.1)


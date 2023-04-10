import cv2
import mediapipe as mp
import tensorflow as tf
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
model_path = r"model\model\training_relu"
model = tf.keras.models.load_model(model_path)
# For webcam input:
#cap = cv2.VideoCapture(0)
import time
import threading
import nep
import numpy as np
import sys
import json
import copy
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

def process_landmarks(landmarks):
  landmarkposition_array = np.empty((33,3))
  for index, landmarkposition in enumerate(landmarks.landmark):
    landmarkposition_array[index][0] = copy.deepcopy(landmarkposition.x)
    landmarkposition_array[index][1] = copy.deepcopy(landmarkposition.y)
    landmarkposition_array[index][2] = copy.deepcopy(landmarkposition.z)
    #print(landmarks.landmark[9].x, landmarkposition_array[9][0])
  for i in range(33):
    landmarkposition_array[i][0] -= landmarkposition_array[0][0]
    landmarkposition_array[i][1] -= landmarkposition_array[0][1]
    landmarkposition_array[i][2] -= landmarkposition_array[0][2]
    
  landmarkposition_array_1_dimension = np.empty(75)
  for i in range(25):
    for j in range(3):
      landmarkposition_array_1_dimension[i*3+j] = landmarkposition_array[i][j]
  
  def maximum(array):
        max = 0
        for i in range(75):
            if max < array[i]:
                max = array[i]
        return max
    
  max_length = maximum(landmarkposition_array_1_dimension)
  landmarkposition_array_1_dimension /= max_length
  return landmarkposition_array_1_dimension

node = nep.node('cropped_image_listener')
conf = node.hybrid("127.0.0.1") # change the IP displayed when using nep master wifi
sub_image = node.new_sub('cropped_image','image',conf)
x = threading.Thread(target=thread_function, args=(1,))
x.start()
print("hoge")
time.sleep(3)
node2 = nep.node('pose_position_pub')
conf2 = node2.hybrid("127.0.0.1")
pub_list = node2.new_pub('75_points', 'json', conf)
list_arange = np.arange(75).tolist()
point_position = np.zeros(75)
csv_path = r"presentation\model\data.csv"
csv_y_path = r"presentation\model\data_y.csv"
processed_landmarks_csv_formed = np.zeros(75).reshape(1,75)
open_close_array = np.array([0])
mode = None
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while True:
    #if img == None:
    #  continue
    image = img
    #if not success:
    #  print("Ignoring empty camera frame.")
    #  # If loading a video, use 'break' instead of 'continue'.
    #  continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #for index, xyz in enumerate(results.pose_landmarks.landmark):
    #  point_position[index * 3 + 0] = xyz.x
    #  point_position[index * 3 + 1] = xyz.y
    #  point_position[index * 3 + 2] = xyz.z
    #  print(index)
    if results.pose_landmarks is not None:
      processed_landmarks = process_landmarks(results.pose_landmarks)
      list_position = processed_landmarks.tolist()
      position_dict = dict(zip(list_arange,list_position))
      sys.stdout.flush()
      pub_list.publish(position_dict)
      key = cv2.waitKey(1)
      processed_landmarks = processed_landmarks.reshape(1,75)
      predict_result = model.predict(processed_landmarks)
      print(np.squeeze(predict_result))
      print("predicted: ",np.argmax(np.squeeze(predict_result)))
      time.sleep(.1)


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
      break

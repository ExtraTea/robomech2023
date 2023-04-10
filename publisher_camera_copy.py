


import cv2
import numpy as np
import sys
import nep
import time


id_ = 0
x_resolution = 1280
y_resolution = 720


node = nep.node('camera_web_p3y')
#conf = node.direct(ip = "192.168.1.6", port = 3000, mode="one2many") 
conf = node.hybrid("192.168.128.127") # change the IP displayed when using nep master wifi
pub_image = node.new_pub('cropped_image','image',conf)


try:

    print ("waiting service")
    sys.stdout.flush()
    video = cv2.VideoCapture(id_)
    print ("service started")
    sys.stdout.flush()

    video.set(cv2.CAP_PROP_FRAME_WIDTH, x_resolution)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, y_resolution)
    time.sleep(1)



    while True:
        success, frame = video.read()
        pub_image.publish(frame)
        time.sleep(.001)


except:
    # Used in the interface - not erase
    video.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    pass
    



import cv2
import numpy as np
from model import asl_model

path_json = '../squeezenet/squeeze_asl.json'
path_weights = '../squeezenet/best_weights_squeeze.h5'

model = asl_model(path_json, path_weights)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):

        font_type = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left = (10,400)
        font_size = 1
        text_colour = (0,255,0)
        line_type = 2


        success, frame = self.video.read()
        # image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        # 	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # 	break

        # Horizontally flipping the video frame to mirror our actions (easier for the user to orient themselves)
        frame2=cv2.flip(frame, 1)

        # Selecting a 150x150 pixel window towards the right of the frame to capture the user's right hand  
        frame3 = frame2[100:300, 440:640]

        # Resizing based on model input shape (200x200), (150x150), (64x64)
        #frame3 = cv2.resize(frame3, (200, 200))

        # Create a mask 2 pixels larger in height and width as per cv2.floodFill documentation
        # Outlining the hand
        detect_edge = cv2.Canny(frame3, 100, 100)
        # Adding 2 pixels in height and width
        edge_mask = cv2.copyMakeBorder(detect_edge,1,1,1,1,cv2.BORDER_REFLECT)

        # Create a copy of 'frame3' to floodFill
        to_model = frame3.copy()

        # Replace all neighbouring pixels that have similar RGB values as the 'seedPoint' pixel (starting point)
        # with white, 'newVal' of (255,255,255)
        cv2.floodFill(image=to_model, mask=edge_mask, seedPoint=(3,3), newVal=(255,255,255), 
                        loDiff=(2,151,100), upDiff=(2,151,100), flags=8)
        # Initiating at the other corners of the image
        cv2.floodFill(to_model, edge_mask, (199,2), (255,255,255), (2,151,65), (2,151,65), flags=8)
        cv2.floodFill(to_model, edge_mask, (199,197), (255,255,255), (2,151,65), (2,151,65), flags=8)
        cv2.floodFill(to_model, edge_mask, (2,197), (255,255,255), (2,151,100), (2,151,100), flags=8)

        # Will need to add a dimension as model expects (None, 300, 300, 3)
        frame4 = np.expand_dims(to_model, axis=0)

        prediction, score = model.predict_image(frame4)

        cv2.rectangle(frame2,(440,100),(640,300),(0,255,0),4)

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Result: {prediction}, Score: {score}%', bottom_left, font_type, font_size, text_colour, line_type)


        ret, jpeg = cv2.imencode('.jpg', frame2)
        return jpeg.tobytes()

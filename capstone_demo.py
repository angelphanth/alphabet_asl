import cv2, time
import numpy as np 
import pandas as pd 
from keras.models import load_model, model_from_json

# Creating a list of the classes 
LETTER_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
      'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Loading in the model architecture and weights
with open('VGG16/VGG16_all.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load model weights
loaded_model.load_weights('VGG16/best_weights_vgg16.h5')
loaded_model._make_predict_function()

# Creating a function that will feed the processed image into the model 
def predict_image(image):
    '''
    Returns a predicted class for the image and the probability for that class.
    
    INPUTS
    ----
    image: the image to predict
    
    OUTPUTS
    -----
    result: the class with the highest probability
    score: the probability score associated with that class for the image (the highest probability)
    '''    

    # Getting the array of all class probabilities
    probas = loaded_model.predict(image)

    # Determining the index of the highest probability and pulling the associated class from the list of classes ('LETTER_LIST')
    result = LETTER_LIST[np.argmax(probas)]
    
    # Saving the probability as a percentage with 2 decimal points to the variable 'score'
    score = float("%0.2f" % (max(probas[0]) * 100))

    return result, score

# Initializing video capture from webcam 1 (front-facing)
video = cv2.VideoCapture(1)

# Variables for later formatting the text that will be printed on the display
font_type = cv2.FONT_HERSHEY_SIMPLEX
bottom_left = (10,400)
font_size = 1
text_colour = (0,255,0)
line_type = 2

# Setting up a counter that upon exit of the video capture will return the number of predictions made
a=0

while True:

    # Time counter
    a += 1 

    # Saving the boolean to 'ret' and the image array to 'frame
    ret, frame = video.read()

    # Horizontally flipping the video frame to mirror our actions (easier for the user to orient themselves)
    frame2=cv2.flip(frame, 1)

    # Selecting a 150x150 pixel window towards the right of the frame to capture the user's right hand  
    frame3 = frame2[100:400, 340:640]

    # Resizing based on model input shape (200x200), (150x150), (64x64)
    frame3 = cv2.resize(frame3, (200, 200))

    # Can brighten and normalize the array if model was trained on asl_colour.csv or asl_original.csv
    #brighter = brightening(frame3, 3)

    # If brightening, array *255 to integer to allow for display 
    #frame3 = (brighter*255).astype('uint8')

    # Will need to add a dimension as model expects (None, 200, 200, 3) for example
    # If brightening the image then input 'brighter', if not then input 'frame3'
    frame4 = np.expand_dims(frame3, axis=0)
    
    # Use the predict_image function to get prediction and associated probability score
    prediction, score = predict_image(frame4)

    # Adding text to the flipped frame
    cv2.putText(frame2, f'Result: {prediction}, Score: {score}%', bottom_left, font_type, font_size, text_colour, line_type)

    # Displaying the frames 
    # Entire frame (flipped for easier use)
    cv2.imshow("Say cheese!", frame2)
    # The image that gets "sent" to the model (prior to adding a dimension)
    cv2.imshow("To the model!", frame3)

    # Printing the results in the command line window
    print(f'Result: {prediction}, Score: {score}')

    # Pressing the letter 'q' will break the while loop so script can proceed
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

# Printing the number of predictions that were made
print(a)

# Exiting video capture and ridding of the displays
video.release()
cv2.destroyAllWindows()
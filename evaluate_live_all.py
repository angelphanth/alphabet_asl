# Written for 640 x 480 video capture

import cv2, time
import numpy as np 
import pandas as pd 
from keras.models import load_model, model_from_json

# To evaluate the models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

# Creating axis labels for future heatmaps

# Creating a list of the classes 
LETTER_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
      'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
# Preficing classes with 'Predicted ' 
predicted_labels = list('Predicted '  + letter for letter in LETTER_LIST)
# Preficing classes with 'True '
true_labels = list('True ' + letter for letter in LETTER_LIST)

# Loading in all the model architecture and weights

# Starting with Model used for Facial Expression Recognition from live video without transformations
with open('CNN_rawlive/CNN_allasl_200_arch.json', "r") as json_file:
    noPre_model_json = json_file.read()
    noPre_model = model_from_json(noPre_model_json)

    # Load model weights
    noPre_model.load_weights('CNN_rawlive/best_weights_all200.h5')
    noPre_model._make_predict_function()

# The face CNN with transformations 
with open('CNN_rawlive_transform/CNN_allasl_200_arch_transform.json', "r") as json_file:
    live_model_json = json_file.read()
    live_model = model_from_json(live_model_json)

    # Load model weights
    live_model.load_weights('CNN_rawlive_transform/best_weights_all200_transform.h5')
    live_model._make_predict_function()
    
# VGG16
with open('VGG16/VGG16_all.json', "r") as json_file:
    vgg16_model_json = json_file.read()
    vgg16_model = model_from_json(vgg16_model_json)

    # Load model weights
    vgg16_model.load_weights('VGG16/best_weights_vgg16.h5')
    vgg16_model._make_predict_function()

# VGG19
with open('VGG19/VGG19_all.json', "r") as json_file:
    vgg19_model_json = json_file.read()
    vgg19_model = model_from_json(vgg19_model_json)

    # Load model weights
    vgg19_model.load_weights('VGG19/best_weights_vgg19.h5')
    vgg19_model._make_predict_function()

# Xception
with open('xception/xception.json', "r") as json_file:
    xception_model_json = json_file.read()
    xception_model = model_from_json(xception_model_json)

    # Load model weights
    xception_model.load_weights('xception/best_weights_xception.h5')
    xception_model._make_predict_function()
    
# InceptionResNetV2
with open('Inception_capstone/inception_capstone.json', "r") as json_file:
    inception_model_json = json_file.read()
    inception_model = model_from_json(inception_model_json)

    # Load model weights
    inception_model.load_weights('Inception_capstone/best_weights_inception_capstone.h5')
    inception_model._make_predict_function()

# Creating a function that will feed the processed image into a chosen model 
def predict_image(model, image):
    '''
    Returns a predicted class for the image and the probability for that class.
    
    INPUTS
    ----
    model: the model to make a prediction 
    image: the image to predict
    
    OUTPUTS
    -----
    result: the class with the highest probability
    score: the probability score associated with that class for the image (the highest probability)
    '''    

    # Getting the array of all class probabilities
    probas = model.predict(image)

    # Choosing the class with the highest probability
    result = np.argmax(probas)
    
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

# Creating lists for each model
# no_pre_model
noPre_predictions = []
noPre_scores = []
# live_model
live_predictions = []
live_scores = []
# vgg16_model
vgg16_predictions = []
vgg16_scores = []
# vgg19_model
vgg19_predictions = []
vgg19_scores = []
# xception_model
xception_predictions = []
xception_scores = []
# inception_model
inception_predictions = []
inception_scores = []

# Creating list of true classes
true_letters = []

# Creating counters for each class 
for n in range(0,29):
    globals()['class_' + str(n)] = 0

# Setting up a counter to keep track of frames read
a=0

# Getting the predictions
while True:

    # Frame counter
    a += 1 

    # Saving the boolean to 'ret' and the image array to 'frame
    ret, frame = video.read()

    # Horizontally flipping the video frame to mirror our actions (easier for the user to orient themselves)
    frame2=cv2.flip(frame, 1)

    # Selecting a 200x200 pixel window towards the right of the frame to capture the user's right hand  
    frame3 = frame2[100:400, 340:640]

    # Resizing based on model input shape (200x200), (150x150), (64x64)
    frame3 = cv2.resize(frame3, (200, 200))

    # Will need to add a dimension as model expects (None, 200, 200, 3) for example
    frame4 = np.expand_dims(frame3, axis=0)
    
    # Use the predict_image function to get prediction and associated probability score
    # First noPre_model
    noPre_predict, noPre_score = predict_image(noPre_model, frame4)
    # live_model
    live_predict, live_score = predict_image(live_model, frame4)
    # vgg16_model
    vgg16_predict, vgg16_score = predict_image(vgg16_model, frame4)
    # vgg_19
    vgg19_predict, vgg19_score = predict_image(vgg19_model, frame4)
    # xception_model
    xception_predict, xception_score = predict_image(xception_model, frame4)
    # inception_model
    inception_predict, inception_score = predict_image(inception_model, frame4)

    # Getting live testset of class 0 (A)
    if a <= 10:

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "A" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 10) & (a <=30):
        # counter for class 0 
        class_0 += 1

        # Appending true_class
        true_letters.append(0)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "A" {class_0}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 1 (B)
    elif (a > 30) & (a<=40):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "B" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 40) & (a <=60):
        # counter for class 0 
        class_1 += 1

        # Appending true_class
        true_letters.append(1)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "B" {class_1}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 2 (C)
    elif (a > 60) & (a<=70):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "C" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 70) & (a <=90):
        # counter for class 0 
        class_2 += 1

        # Appending true_class
        true_letters.append(2)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "C" {class_2}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 3 (D)
    elif (a > 90) & (a<=100):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "D" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 100) & (a <=120):
        # counter for class 0 
        class_3 += 1

        # Appending true_class
        true_letters.append(3)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "D" {class_3}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 4 (E)
    elif (a > 120) & (a<=130):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "E" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 130) & (a <=150):
        # counter for class 0 
        class_4 += 1

        # Appending true_class
        true_letters.append(4)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "E" {class_4}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 5 (F)
    elif (a > 150) & (a<=160):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "F" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 160) & (a <=180):
        # counter for class 0 
        class_5 += 1

        # Appending true_class
        true_letters.append(5)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "F" {class_5}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 6 (G)
    elif (a > 180) & (a<=190):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "G" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 190) & (a <=210):
        # counter for class 0 
        class_6 += 1

        # Appending true_class
        true_letters.append(6)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "G" {class_6}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 7 (H)
    elif (a > 210) & (a<=220):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "H" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 220) & (a <=240):
        # counter for class 0 
        class_7 += 1

        # Appending true_class
        true_letters.append(7)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "H" {class_7}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 8 (I)
    elif (a > 240) & (a<=250):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "I" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 250) & (a <=270):
        # counter for class 0 
        class_8 += 1

        # Appending true_class
        true_letters.append(8)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "I" {class_8}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 9 (J starting position)
    elif (a > 270) & (a<=280):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "J" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 280) & (a <=300):
        # counter for class 0 
        class_9 += 1

        # Appending true_class
        true_letters.append(9)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "J" {class_9}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 10 (K)
    elif (a > 300) & (a<=310):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "K" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 310) & (a <=330):
        # counter for class 0 
        class_10 += 1

        # Appending true_class
        true_letters.append(10)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "K" {class_10}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 11 (L)
    elif (a > 330) & (a<=340):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "L" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 340) & (a <=360):
        # counter for class 0 
        class_11 += 1

        # Appending true_class
        true_letters.append(11)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "L" {class_11}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 12 (M)
    elif (a > 360) & (a<=370):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "M" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 370) & (a <=390):
        # counter for class 0 
        class_12 += 1

        # Appending true_class
        true_letters.append(12)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "M" {class_12}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 13 (N)
    elif (a > 390) & (a<=400):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "N" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 400) & (a <=420):
        # counter for class 0 
        class_13 += 1

        # Appending true_class
        true_letters.append(13)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "N" {class_13}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 14 (O)
    elif (a > 420) & (a<=430):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "O" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 430) & (a <=450):
        # counter for class 0 
        class_14 += 1

        # Appending true_class
        true_letters.append(14)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "O" {class_14}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 15 (P)
    elif (a > 450) & (a<=460):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "P" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 460) & (a <=480):
        # counter for class 0 
        class_15 += 1

        # Appending true_class
        true_letters.append(15)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "P" {class_15}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 16 (Q)
    elif (a > 480) & (a<=490):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "Q" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 490) & (a <=510):
        # counter for class 0 
        class_16 += 1

        # Appending true_class
        true_letters.append(16)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "Q" {class_16}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 17 (R)
    elif (a > 510) & (a<=520):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "R" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 520) & (a <=540):
        # counter for class 0 
        class_17 += 1

        # Appending true_class
        true_letters.append(17)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "R" {class_17}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 18 (S)
    elif (a > 540) & (a<=550):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "S" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 550) & (a <=570):
        # counter for class 0 
        class_18 += 1

        # Appending true_class
        true_letters.append(18)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "S" {class_18}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 19 (T)
    elif (a > 570) & (a<=580):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "T" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 580) & (a <=600):
        # counter for class 0 
        class_19 += 1

        # Appending true_class
        true_letters.append(19)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "T" {class_19}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 20 (U)
    elif (a > 600) & (a<=610):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "U" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 610) & (a <=630):
        # counter for class 0 
        class_20 += 1

        # Appending true_class
        true_letters.append(20)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "U" {class_20}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 21 (V)
    elif (a > 630) & (a<=640):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "V" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 640) & (a <=660):
        # counter for class 0 
        class_21 += 1

        # Appending true_class
        true_letters.append(21)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "V" {class_21}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 22 (W)
    elif (a > 660) & (a<=670):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "W" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 670) & (a <=690):
        # counter for class 0 
        class_22 += 1

        # Appending true_class
        true_letters.append(22)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "W" {class_22}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 23 (X)
    elif (a > 690) & (a<=700):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "X" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 700) & (a <=720):
        # counter for class 0 
        class_23 += 1

        # Appending true_class
        true_letters.append(23)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "X" {class_23}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 24 (Y)
    elif (a > 720) & (a<=730):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "Y" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 730) & (a <=750):
        # counter for class 0 
        class_24 += 1

        # Appending true_class
        true_letters.append(24)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "Y" {class_24}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 25 (Z starting position)
    elif (a > 750) & (a<=760):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "Z" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 760) & (a <=780):
        # counter for class 0 
        class_25 += 1

        # Appending true_class
        true_letters.append(25)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "Z" {class_25}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 26 (del)
    elif (a > 780) & (a<=790):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "delete" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 790) & (a <=810):
        # counter for class 0 
        class_26 += 1

        # Appending true_class
        true_letters.append(26)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "del" {class_26}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 27 (nothing)
    elif (a > 810) & (a<=820):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get ready to do nothing!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 820) & (a <=840):
        # counter for class 0 
        class_27 += 1

        # Appending true_class
        true_letters.append(27)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "nothing" {class_27}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Getting live testset of class 28 (space)
    elif (a > 840) & (a<=850):

        # Adding text to the flipped frame
        cv2.putText(frame2, f'Get your "space" ready!', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    elif (a > 850) & (a <=870):
        # counter for class 0 
        class_28 += 1

        # Appending true_class
        true_letters.append(28)

        # Appending the predictions to their lists 
        # noPre_model
        noPre_predictions.append(noPre_predict)
        noPre_scores.append(noPre_score)
        # live_model
        live_predictions.append(live_predict)
        live_scores.append(live_score)
        # vgg16_model
        vgg16_predictions.append(vgg16_predict)
        vgg16_scores.append(vgg16_score)
        # vgg19_model
        vgg19_predictions.append(vgg19_predict)
        vgg19_scores.append(vgg19_score)
        # xception_model
        xception_predictions.append(xception_predict)
        xception_scores.append(xception_score)
        # inception_model
        inception_predictions.append(inception_predict)
        inception_scores.append(inception_score)

        # Adding text to the flipped frame
        cv2.putText(frame2, f' "space" {class_28}/20', bottom_left, font_type, font_size, text_colour, line_type)

        # Displaying the frames 
        # Entire frame (flipped for easier use)
        cv2.imshow("Say cheese!", frame2)
        # The image that gets "sent" to the model (prior to adding a dimension)
        cv2.imshow("To the model!", frame3)

    # Pressing the letter 'q' will break the while loop so script can proceed
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

    elif a > 870:
        break

# Exiting video capture and ridding of the displays
video.release()
cv2.destroyAllWindows()

# Printing the number of frames that were captured and lists
print(f'Frames captured: {a}')

# Save predictions to a df
prediction_summary = pd.DataFrame([true_letters, noPre_predictions, live_predictions, vgg16_predictions, 
vgg19_predictions, xception_predictions, inception_predictions], index=['True Class','noPre_model','live_model',
'vgg16_model','vgg19_model','xception_model','inception_model'])
# Save the df to a csv file 
prediction_summary.to_csv('live_predictions/prediction_summary.csv')
# Display the df
prediction_summary


# Print update
print('Creating confusion matrices and classification reports for all models.')


# First noPre_model

# Getting accuracy
print(f'noPre_model accuracy = {accuracy_score(true_letters, noPre_predictions)} \n')

# Displaying the classification report for the validation set
print('noPre_model Classification Report\n \n', classification_report(true_letters, noPre_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
noPre_matrix = pd.DataFrame(confusion_matrix(true_letters, noPre_predictions), columns=predicted_labels, index=true_labels)
# Show it 
noPre_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(noPre_matrix/noPre_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: noPre_model', fontsize=20)
plt.savefig('live_predictions/noPre_matrix.png')

# live_model

# Getting accuracy
print(f'live_model accuracy = {accuracy_score(true_letters, live_predictions)} \n')

# Displaying the classification report for the validation set
print('live_model Classification Report\n \n', classification_report(true_letters, live_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
live_matrix = pd.DataFrame(confusion_matrix(true_letters, live_predictions), columns=predicted_labels, index=true_labels)
# Show it 
live_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(live_matrix/live_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: live_model', fontsize=20)
plt.savefig('live_predictions/live_matrix.png')


# vgg16_model

# Getting accuracy
print(f'vgg16_model accuracy = {accuracy_score(true_letters, vgg16_predictions)} \n')

# Displaying the classification report for the validation set
print('vgg16_model Classification Report\n \n', classification_report(true_letters, vgg16_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
vgg16_matrix = pd.DataFrame(confusion_matrix(true_letters, vgg16_predictions), columns=predicted_labels, index=true_labels)
# Show it 
vgg16_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(vgg16_matrix/vgg16_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: vgg16_model', fontsize=20)
plt.savefig('live_predictions/vgg16_matrix.png')


# vgg19_model

# Getting accuracy
print(f'vgg19_model accuracy = {accuracy_score(true_letters, vgg19_predictions)} \n')

# Displaying the classification report for the validation set
print('vgg19_model Classification Report\n \n', classification_report(true_letters, vgg19_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
vgg19_matrix = pd.DataFrame(confusion_matrix(true_letters, vgg19_predictions), columns=predicted_labels, index=true_labels)
# Show it 
vgg19_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(vgg19_matrix/vgg19_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: vgg19_model', fontsize=20)
plt.savefig('live_predictions/vgg19_matrix.png')


# xception_model

# Getting accuracy
print(f'xception_model accuracy = {accuracy_score(true_letters, xception_predictions)} \n')

# Displaying the classification report for the validation set
print('xception_model Classification Report\n \n', classification_report(true_letters, xception_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
xception_matrix = pd.DataFrame(confusion_matrix(true_letters, xception_predictions), columns=predicted_labels, index=true_labels)
# Show it 
xception_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(xception_matrix/xception_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: xception_model', fontsize=20)
plt.savefig('live_predictions/xception_matrix.png')


# inception_model

# Getting accuracy
print(f'inception_model accuracy = {accuracy_score(true_letters, inception_predictions)} \n')

# Displaying the classification report for the validation set
print('inception_model Classification Report\n \n', classification_report(true_letters, inception_predictions, target_names=LETTER_LIST),'\n')

# Creating a non-normalized confusion matrix for the validation set
inception_matrix = pd.DataFrame(confusion_matrix(true_letters, inception_predictions), columns=predicted_labels, index=true_labels)
# Show it 
inception_matrix

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(inception_matrix/inception_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: inception_model', fontsize=20)
plt.savefig('live_predictions/inception_matrix.png')
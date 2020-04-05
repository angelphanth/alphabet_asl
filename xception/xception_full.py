#!/usr/bin/env python
# coding: utf-8

# To create callbacks
#from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# importing layers
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report 

# Saving variables that will be called upon throughout the script 
classes = 29 
pixels = 200
channels = 3 #if 3 'color_mode="rgb"', if 1 'color_mode="grayscale"'
path = '../brighten_only/'
batch_size = 20
epochs=20

# Initialize the model 
model = Sequential() 

# Import the pretrained VGG19 network, do not include the top layers
# Import the architecture with the weights by ='imagenet'
pretrained = Xception(weights='imagenet', include_top=False, pooling='max', input_shape=(pixels,pixels,channels))

# Display the model
pretrained.summary()

# setting them to not trainable so weights wont be tweaked
for layer in pretrained.layers[:33]:
    layer.trainable=False
    
for layer in pretrained.layers[33:]:
    layer.trainable=True

# Adding the pretrained model
model.add(pretrained)
# Adding additional layers
model.add(Dense(classes, activation='softmax'))
    
# Changing image input_shape format
#from keras import backend  
#backend.set_image_data_format('channels_last')
    
# Istantiating Adam optimizer with a learning rate of 0.0001 and saving to variable 'optim'
optim = Adam(lr=0.0001)

# Compiling the CNN model 
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary 
model.summary()

# Saving the model summary as an image
plot_model(model, to_file='inception_resnetv2.png', 
show_shapes=True, show_layer_names=True)                          


# Istantiating the training/validation ImageDataGenerator()
datagen_train = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, 
                                   width_shift_range=[-5,5],
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   fill_mode='reflect',
                                   validation_split=0.1)

# Istantiating the test generator
datagen_test = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True, 
                                   width_shift_range=[-5,5],
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   fill_mode='reflect')

# The generator for the training data, shuffle set to True 
# (unable to view confusion matrix and classification report as a result)
train_generator = datagen_train.flow_from_directory(path+'training_set/',
                                                    target_size=(pixels, pixels),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True, subset='training')

# Generator for validation data, shuffle set to 'False' to allow for model evaluation
validation_generator = datagen_train.flow_from_directory(path+'training_set/',
                                                    target_size=(pixels, pixels),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False, subset='validation')

# Generator for test data, shuffle set to 'False' to allow for model evaluation 
test_generator = datagen_test.flow_from_directory(path+'test_set/',
                                                    target_size=(pixels, pixels),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Saving the number of stepsizes for the training, validation and test sets 
train_stepsize = train_generator.n//train_generator.batch_size 
valid_stepsize = validation_generator.samples//validation_generator.batch_size 
print(f'Training step size = {train_stepsize}, Validation step size = {valid_stepsize}')
test_stepsize = test_generator.samples//test_generator.batch_size
print(f'Test step size = {test_stepsize}')


# Creating callbacks
# Model checkpointer that will only save the weights from the epoch with the best validation accuracy
checkpoint = ModelCheckpoint("best_weights_xception.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# Early stopping will help prevent overfitting the model to the train data and will also monitor validation accuracy
# Will not stop before 5 epochs (patience)
ES = EarlyStopping(monitor='val_accuracy', mode='auto', min_delta=0, patience=5, verbose=1)
# Adding both callbacks to a 'callbacks_list'
callbacks_list = [checkpoint, ES]    


# Fitting the model to the training data
# Saving the progress to variable 'history' to allow for plotting of train and validation accuracies and losses 
history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_stepsize,
                                epochs=epochs,
                                validation_data=validation_generator,
                                validation_steps=valid_stepsize,
                                callbacks=callbacks_list)


# Plotting the model's test and validation loss progression for every completed epoch 
plt.subplots(1,2,figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam , Learning Rate: 0.0001', fontsize=20)
plt.ylabel('Loss', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='^', color='red')
plt.legend(loc='upper right')
# Plotting the test and validation accuracy changes for every epoch 
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='^', color='red')
plt.legend(loc='lower right')
plt.show()

# Saving the CNN architecture as a JSON
modelarch_json = model.to_json()
with open("xception.json","w") as json_file:
    json_file.write(modelarch_json)
    
# Initialize the 'bestmodel' using the weights saved with the ModelCheckpoint callback
bestmodel = model_from_json(modelarch_json)
# Loading the weights that resulted in the best validation accuracy during training 
bestmodel.load_weights("best_weights_xception.h5")
# Compile the bestmodel
bestmodel.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# Creating axis labels for future heatmaps 
# Getting the class names
categories = list(train_generator.class_indices.keys())
# Preficing classes with 'Predicted ' 
predicted_labels = list('Predicted '  + category for category in categories)
# Preficing classes with 'True '
true_labels = list('True ' + category for category in categories)

# Creating labels(indices) for model evaluation metrics 
eval_metrics = list(bestmodel.metrics_names)

# NOTE: Unable to create a classification report or confusion matrix for train data as
# train_generator(shuffle=True) does not save the true classes 

# Re-evaluating the best model's performance on the training data
train_eval = bestmodel.evaluate_generator(train_generator, steps=train_stepsize)
# Saving the results to a dataframe
df_train_eval = pd.DataFrame(list(train_eval), index=eval_metrics, columns=['Train'])
# Display the results
print(df_train_eval)


# Getting bestmodel's predictions (as probabilities) on the validation images 
validation_probas = bestmodel.predict_generator(validation_generator, steps=valid_stepsize)
# Setting the model's class prediction as the class that received the highest probability for each image
valid_predictions = validation_probas.argmax(axis=1)

# Getting the true class labels for the validation data
valid_true = validation_generator.classes

# Evaluating bestmodel's performance on the validation data 
valid_eval = bestmodel.evaluate_generator(validation_generator, steps=valid_stepsize)
# Saving the results to a dataframe
df_valid_eval = pd.DataFrame(list(valid_eval), index=eval_metrics, columns=['Validation'])
# Display the results 
print(df_valid_eval)

# Displaying the classification report for the validation set
print('Validation Classification Report\n \n', classification_report(valid_true, valid_predictions, target_names=categories))

# Creating a non-normalized confusion matrix for the validation set
valid_matrix = pd.DataFrame(confusion_matrix(valid_true, valid_predictions), columns=predicted_labels, index=true_labels)

# Plotting the normalized confusion matrix (proportion of predictions by class) as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(valid_matrix/valid_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: Validation Data', fontsize=20)
plt.show()


# Getting bestmodel's predictions (as probabilities) on the test set 
test_probas = bestmodel.predict_generator(test_generator, steps=test_stepsize)
# Setting the model's class prediction as the class that received the highest probability for each image
test_predictions = test_probas.argmax(axis=1)

# Getting the true class labels for the test set
test_true = test_generator.classes

# Evaluating the best model's performance on the test data 
test_eval = bestmodel.evaluate_generator(test_generator, steps=test_stepsize)\
# Saving the results to a dataframe 
df_test_eval = pd.DataFrame(list(test_eval), index=eval_metrics, columns=['Test'])
# Display the results
print(df_test_eval)

# Displaying the classification report for the test set
print('Test Classification Report\n \n', classification_report(test_true, test_predictions, target_names=categories))

# Non-normalized confusion matrix for the test data
test_matrix = pd.DataFrame(confusion_matrix(test_true, test_predictions), columns=predicted_labels, index=true_labels)

# Plotting a normalized confusion matrix for the test data as a heatmap 
sns.set(font_scale=1.2)
plt.figure(figsize=(15,15))
sns.heatmap(round(test_matrix/test_matrix.sum(axis=1)[:,np.newaxis],2), cmap='Blues', annot=True, linewidths=.5, linecolor='black')
plt.title('Normalized Confusion Matrix: Test Data', fontsize=20)
plt.show()
from tensorflow.keras.models import model_from_json
import numpy as np

class asl_model(object):

    LETTER_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
      'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_image(self, img):
        preds = self.loaded_model.predict(img)
        # Determining the index of the highest probability and pulling the associated class from the list of classes ('LETTER_LIST')
        self.result = asl_model.LETTER_LIST[np.argmax(preds)]
    
        # Saving the probability as a percentage with 2 decimal points to the variable 'score'
        self.score = float("%0.2f" % (max(preds[0]) * 100))
        return self.result, self.score
import numpy as np
import csv
import sys
import json

from validate import validate
from preprocessor import preprocessing
"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    with open(test_X_file_path, newline='') as f:
        reader = csv.reader(f)
        tempX = list(reader)
    test_X=[]
    for l in tempX:
        test_X.append("".join(l))
    
    with open(model_file_path,'r') as f:
        model = json.load(f)
    return test_X, model


def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators):
    attribute_P = []
    textc = class_wise_frequency_dict[c]
    D = class_wise_denominators[c]
    for w in test_X.split():
        if w in textc.keys():
            attribute_P.append(np.log((textc[w]+1)/D))
        else:
            attribute_P.append(np.log(1/D))
        
    return sum(attribute_P)


def predict(test_X, prior, frequency_dict, denominators):
    class_posterior = dict()
    classes = list(prior.keys())
    for c in classes:
        class_posterior[c] = np.log(prior[c]) + compute_likelihood(test_X, c, frequency_dict, denominators)
    return list(class_posterior.keys())[list(class_posterior.values()).index(max(class_posterior.values()))]


def predict_target_values(test_Xs, model):
    predY = []
    freq_dict = model['cwwfd']
    denominator = model['cwd']
    prior = model['pp']
    for t_X in test_Xs:
        predY.append(predict(t_X, prior, freq_dict,denominator))
    return np.array(predY)
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predictY(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, "./MODEL_FILE.json")
    test_X = preprocessing(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predictY(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 

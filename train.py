import csv
import json
import numpy as np
from preprocessor import preprocessing
from sklearn.model_selection import train_test_split

def import_data():
    with open('train_X_nb.csv', newline='') as f:
        reader = csv.reader(f)
        tempX = list(reader)
    X=[]
    for l in tempX:
        X.append("".join(l))
    Y = np.genfromtxt("train_Y_nb.csv",delimiter=',')
    return X, Y

def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict = dict()
    for i in range(len(X)):
        words = X[i].split()
        for token_word in words:
            y = Y[i]
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[int(y)] = dict()
            if token_word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[int(y)][token_word] = 0
            class_wise_frequency_dict[int(y)][token_word] += 1
    return class_wise_frequency_dict


def get_class_wise_denominators_likelihood(X, Y):
    V = len(list(set((" ".join(X)).split())))

    class_deno_dict = dict()
    class_word_freq_dict = class_wise_words_frequency_dict(X,Y)
    for c in list(set(Y)):
        class_deno_dict[int(c)] = sum(class_word_freq_dict[c].values()) + V
    #print(class_deno_dict.keys())
    return class_deno_dict
    

def compute_prior_probabilities(Y):
    class_prior_prob = dict()
    for c in Y:
        if c not in class_prior_prob.keys():
            class_prior_prob[int(c)] = 0
        class_prior_prob[int(c)] += 1/len(Y)
    #print(class_prior_prob)
    return class_prior_prob


def train(X,Y):
    model = dict()
    model['cwwfd'] = class_wise_words_frequency_dict(X, Y)
    model['cwd'] = get_class_wise_denominators_likelihood(X, Y)
    model['pp'] = compute_prior_probabilities(Y)    
    return model
    

def save_model(model, model_file_name):
    with open(model_file_name, 'w') as model_file:
        json.dump(model,model_file)


if __name__ == "__main__":
    X,Y = import_data()
    X = preprocessing(X)
    #train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 4291/4296)
    #print(train_X,test_Y)
    model = train(X, Y)
    save_model(model,"MODEL_FILE.json")

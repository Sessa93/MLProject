# Author Andrea Sessa
# Perform a Forward Features Selection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

def transform(data, features):
    copy = data
    np.delete(copy, features, axis=1)
    return copy

def evaluate(features, train_x, train_t):
    svc = svm.SVC()
    svc.fit(transform(train_x, features), train_t)
    return svc.score(transform(train_x, features), train_t)

def fws(tol, features, train_x, train_t):
    current_s = []
    last = -1
    max_score = 0
    while max_score-last > tol:
        last = max_score
        max_score = evaluate(current_s + [features[0]], train_x, train_t)
        best_f = features[0]
        for f in features[1:]:
            curr_score = evaluate(current_s + [f], train_x, train_t)
            if curr_score > max_score:
                best_f = f
                max_score = curr_score
        current_s.append(best_f)
        print("Included: "+str(best_f)+ " Score: "+str(max_score))
        features.remove(best_f)
    return current_s

def main():
    tol = 0.01

    # Load the data
    data = sio.loadmat('baseline2.mat')
    n_train = int(data['n_train'])
    n_test = int(data['n_test'])
    train_x = np.array(data['train_x'])
    train_t = np.array(data['train_t']).reshape(n_train)
    test_x = np.array(data['test_x'])
    test_t = np.array(data['test_t']).reshape(800)

    # Start forward features selection
    sel_features = fws(tol, range(0,1000), train_x, train_t)

    # Determine the hyperparameters



if __name__ == "__main__":
    main()

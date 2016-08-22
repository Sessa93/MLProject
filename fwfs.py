# Author Andrea Sessa
# Perform a Forward Features Selection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets, feature_selection, cross_validation

def transform(data, features):
    return data[:, features]

def evaluate(features, train_x, train_t):
    svc = svm.SVC(max_iter=1000000)
    svc.fit(transform(train_x, features), train_t)
    this_scores = cross_validation.cross_val_score(svc, transform(train_x, features), train_t, n_jobs=-1, cv=5, scoring='accuracy')
    return sum(this_scores)/len(this_scores)

def fws(tol, features, train_x, train_t):
    current_s = []
    last = -1
    max_score = 0
    while (max_score-last > tol and max_score > last) and len(current_s) <= 200:
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
    tol = 0.0001

    # Load the data
    data = sio.loadmat('baseline2.mat')
    n_train = int(data['n_train'])
    n_test = int(data['n_test'])
    train_x = np.array(data['train_x'])
    train_t = np.array(data['train_t']).reshape(n_train)
    test_x = np.array(data['test_x'])
    test_t = np.array(data['test_t']).reshape(800)

    # Start forward features selection
    sel_features = fws(tol, list(range(0,1000)), train_x, train_t)

    # Determine the hyperparameters
    C = [-10,5,10]
    G = [-10,5,10]
    CF = [-10,5,10]
    train = transform(train_x, sel_features)
    max_score = 0

    print("Started Cross-Validation...")
    for c in C:
        for g in G:
            for cf in CF:
                #Find best C, gamma
                svc = svm.SVC(C=2**c, gamma=2**g, coef0=2**cf, degree=3, kernel='poly',max_iter=1000000)
                this_scores = cross_validation.cross_val_score(svc, train, train_t, n_jobs=-1, cv=5, scoring='accuracy')
                mean_score = sum(this_scores)/len(this_scores)
                print("C: "+str(c)+" G: "+str(g)+" A: "+str(mean_score) + " CF: " +str(cf))
                if mean_score > max_score:
                    max_score = mean_score
                    best_svm = svc
    # Testing
    print("Started testing...")
    test = transform(test_x, sel_features)
    best_svm.fit(train,train_t)
    pred = best_svm.predict(test)
    sio.savemat('predicted_fwfs.mat',dict(x=range(800),pred_t=pred))

    final_score = best_svm.score(test,test_t)
    print(best_svm)
    print("Final Accuracy: "+str(final_score))

if __name__ == "__main__":
    main()

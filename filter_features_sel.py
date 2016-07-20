#Author Andrea Sessa
#Perform a filter feature selection

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn import datasets

def main():
    #Load the dataset from Matlab
    data = sio.loadmat('baseline.mat')
    train_x = np.array(data['train_x'])
    train_t = np.array(data['train_t']).reshape(3200)
    test_x = np.array(data['test_x'])
    test_t = np.array(data['test_t']).reshape(800)

    X_indices = np.arange(train_x.shape[-1])
    print(type(train_t[0]))
    #Plotting
    X_f_scores, X_f_pval = f_classif(train_x, train_t.ravel())
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(X_f_scores)
    ax.set_title('Univariate Feature Selection: Classification F-Score')
    ax.set_xlabel('features')
    ax.set_ylabel('F-score')
    #plt.show()

    #SVM Fitting
    #tuned_parameters = [{'degree': ['3'], 'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
    C = range(-5,25,5)
    G = range(-30,5,-5)
    ###############################################################################
    # Plot the cross-validation score as a function of percentile of features
    scores = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    best_c = 0
    best_g = 0
    max_score = -np.inf

    for p in percentiles:
        anova = feature_selection.SelectPercentile(f_classif, percentile=p)
        anova.fit(train_x,train_t)
        train = anova.transform(train_x)
        test = anova.transform(test_x)
        for c in C:
            for g in G:
                #Find best C, gamma
                svc = svm.SVC(C=2**c, gamma=2**g, degree=3, kernel='poly')
                this_scores = cross_validation.cross_val_score(svc, train, train_t, n_jobs=4, cv=5, scoring='accuracy')
                mean_score = sum(this_scores)/len(this_scores)
                print("C: "+str(c)+" G: "+str(g)+" P: "+str(p)+" A: "+str(mean_score))
                if mean_score > max_score:
                    max_score = mean_score
                    best_c = c
                    best_g = g

        best_svm = svm.SVC(C=2**best_c,degree=3, kernel='poly',gamma=2**best_g)
        acc = cross_validation.cross_val_score(best_svm, test, test_t, n_jobs=1, cv=5,scoring='accuracy')
        mean_acc = sum(acc)/len(acc)
        scores.append(mean_acc)
        print("Acc on T: "+str(mean_acc))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(scores)
    ax.set_title('Univariate Feature Selection: Classification F-Score')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()

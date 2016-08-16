#Author Andrea Sessa
#Perform a PCA analysis

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

from sklearn import datasets

def main():
    #Load the dataset from Matlab
    data = sio.loadmat('baseline2.mat')
    n_train = int(data['n_train'])
    n_test = int(data['n_test'])
    train_x = np.array(data['train_x'])
    train_t = np.array(data['train_t']).reshape(n_train)
    test_x = np.array(data['test_x'])
    test_t = np.array(data['test_t']).reshape(800)
    X_indices = np.arange(train_x.shape[-1])

    #SVM Fitting
    C = [-10,0,5]
    G = [-10,0,10]
    CF = [-10,0,5,10]

    # Plot the cross-validation score as a function of percentile of features
    components = (5,10,20,50,100,200)
    scores = list()
    svcs = list()
    pcas = list()

    for cc in components:
        best_c = 0
        best_g = 0
        best_cf = 0
        best_pca = None
        max_score = -np.inf

        pca = PCA(n_components=cc)
        pca.fit(train_x)
        train = pca.transform(train_x)

        for c in C:
            for g in G:
                for cf in CF:
                    #Find best C, gamma
                    svc = svm.SVC(C=2**c, gamma=2**g, coef0=2**cf, degree=3, kernel='poly',max_iter=1000000)
                    this_scores = cross_validation.cross_val_score(svc, train, train_t, n_jobs=-1, cv=5, scoring='accuracy')
                    mean_score = sum(this_scores)/len(this_scores)

                    print("C: "+str(c)+" G: "+str(g)+" CMPS: "+str(cc)+" A: "+str(mean_score) + " CF: " +str(cf))
                    if mean_score > max_score:
                        max_score = mean_score
                        best_svm = svc
                        best_pca = pca
        svcs.append(best_svm)
        pcas.append(best_pca)
        scores.append(max_score)

    m_ind =  scores.index(max(scores))
    best_s = svcs[m_ind]
    pca = pcas[m_ind]

    # Test final model
    test = pca.transform(test_x)
    train = pca.transform(train_x)
    best_s.fit(train,train_t)
    final_score = best_s.score(test,test_t)
    print(best_s)
    print("Final Accuracy: "+str(final_score))

    # Plot the result
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(components, scores)
    ax.set_title('PCA Analysis')
    ax.set_xlabel('N. Components')
    ax.set_ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()

#Author Andrea Sessa
#Perform a ISOMAP analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import axes3d, Axes3D

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
    C = [-10,5,10]
    G = [-10,5,10]
    CF = [-10,5,10]

    # Plot the cross-validation score as a function of percentile of features
    NG = [10,20,50,100,200]
    components = (10,20,50,100,200)
    scores = list()
    svcs = list()
    isos = list()

    for cc in components:
        for nn in NG:
            best_c = 0
            best_g = 0
            best_cf = 0
            best_iso = None
            max_score = -np.inf

            iso = Isomap(n_components=cc, n_neighbors=nn)
            iso.fit(train_x)
            train = iso.transform(train_x)

            for c in C:
                for g in G:
                    for cf in CF:
                        #Find best C, gamma
                        svc = svm.SVC(C=2**c, gamma=2**g, coef0=2**cf, degree=3, kernel='poly',max_iter=1000000)
                        this_scores = cross_validation.cross_val_score(svc, train, train_t, n_jobs=-1, cv=5, scoring='accuracy')
                        mean_score = sum(this_scores)/len(this_scores)

                        print("C: "+str(c)+" G: "+str(g)+" CMPS: "+str(cc)+" A: "+str(mean_score) + " CF: " +str(cf) + "N: "+str(nn))

                        if mean_score > max_score:
                            max_score = mean_score
                            best_svm = svc
                            best_iso = iso
            svcs.append(best_svm)
            isos.append(best_iso)
            scores.append(max_score)

    m_ind =  scores.index(max(scores))
    best_s = svcs[m_ind]
    iso = isos[m_ind]

    # Test final model
    test = iso.transform(test_x)
    train = iso.transform(train_x)
    best_s.fit(train,train_t)

    pred = best_s.predict(test)
    sio.savemat('predicted_iso.mat',dict(x=range(800),pred_t=pred))

    final_score = best_s.score(test,test_t)
    print(best_s)
    print("Final Accuracy: "+str(final_score))
    print(scores)

if __name__ == "__main__":
    main()

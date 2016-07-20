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
    test_t = np.array(data['test_t'])

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
    tuned_parameters = [{'degree': ['3'], 'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
    ###############################################################################
    # Plot the cross-validation score as a function of percentile of features
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for p in percentiles:
        anova = feature_selection.SelectPercentile(f_classif, percentile=p)
        anova.fit(train_x,train_t)
        # Compute cross-validation score using all CPUs
        train = anova.transform(train_x)

        #Find best C, gamma
        grid = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
        grid.fit(train, train_t)

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)


if __name__ == "__main__":
    main()

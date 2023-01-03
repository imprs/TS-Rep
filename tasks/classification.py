import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score


def fit_svm(features, y, MAX_SAMPLES=10000):
    ''' Grid search param C for the SVM and return the classifier with
        the best score.
    '''
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

def classification(train_data, train_labels, test_data, test_labels):
    ''' Fit SVM on train_data and evaluate on val_test_data. 
        Return val_test accurracy.
    '''
    clf = fit_svm(train_data, train_labels)
    # Evaluate on train_data
    train_pred = clf.predict(train_data)
    train_acc = accuracy_score(train_labels, train_pred)

    # Evaluate on val_test_data
    test_pred = clf.predict(test_data)
    test_acc = accuracy_score(test_labels, test_pred)

    print(f'train_acc: {train_acc} \ntest_acc:{test_acc}')
    return test_acc


def cross_validation(train_data, train_labels, val_test_data, val_test_labels, n_splits=10):
    ''' Fit SVM on the whole dataset using KFold (K=10 default) cross-validation. 
        Return cross_validation accurracy.
    '''
    # Stack train_data and val_test_data, and train_labels and val_test_labels.
    data = np.vstack((train_data, val_test_data))
    labels = np.hstack((train_labels, val_test_labels))

    clf = fit_svm(train_data, train_labels)

    # Perform 10-Fold CrossValidation
    # scores = cross_val_score(clf, data, labels, cv=10)
    cv = KFold(n_splits=n_splits)
    scores = cross_val_score(clf, data, labels, cv=cv) # Use specifically KFold
    cv_acc = scores.mean()
    print(f'cv_acc: {cv_acc}')
    return cv_acc

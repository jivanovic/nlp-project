import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time


def test_classifier(X_train, y_train, X_test, y_test, classifier):
    print("")
    print("===============================================")
    classifier_name = str(type(classifier).__name__)
    print("Testing " + classifier_name)
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    print("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    print("Predicting time {0}s".format(time() - now))

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    print("=================== Results ===================")
    print("            Content Discussion     Greeting     Logistics     Instruction Question     Assignment Question     General Comment     Incomplete/typo     Feedback Discussion Wrap-up     Outside Material     Opening Statement     General Question     Content Question     Emoticon/Non-verbal     Assignment Instructions     Response")
    print("F1       " + str(f1))
    print("Precision" + str(precision))
    print("Recall   " + str(recall))
    print("Accuracy " + str(accuracy))
    print("===============================================")

    return precision, recall, accuracy, f1


def cv(classifier, X_train, y_train):
    X_train = X_train.fillna(0)
    print("===============================================")
    classifier_name = str(type(classifier).__name__)
    now = time()
    print("Crossvalidating " + classifier_name + "...")
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    print("Crosvalidation completed in {0}s".format(time() - now))
    print("Accuracy: " + str(accuracy[0]))
    print("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    print("===============================================")
    return accuracy
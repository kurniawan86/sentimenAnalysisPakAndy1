import pandas as pd
from sklearn.model_selection import train_test_split
from Evaluate import Evalauate
from Classifier import Classifier
import numpy as np

def printErrorIndex(y_pred, y_test):
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            print(i, y_pred[i], y_test[i])
def readData():
    # read from excel
    df = pd.read_excel('Book1.xlsx', sheet_name='Book1')
    # read each column
    X = df['text'].tolist()
    y = df['class'].tolist()
    return X, y

if __name__ == '__main__':
    X,y = readData()
    dataset = [X,y]

    # divide into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print length of train and test
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
    X = X_train + X_test
    y = y_train + y_test
    # print("YYY ", y)

    # test BoW
    from BoW import boW
    bow_train = boW(X)
    X_train = bow_train.vectorizer()

    #varible for save y_predAll
    y_predAll = []

    # test classifier Naive Bayes
    print("Naive Bayes")
    classifierNB = Classifier(X_train, y)
    classifierNB.MultinomialBN()
    y_predNB = classifierNB.MultinomialBN()
    y_predAll.append(y_predNB)
    # test evaluate data Naive Bayes
    evaluate = Evalauate(y_predNB, y)
    # print(np.array(y_predNB).transpose())
    # print(np.array(y).transpose())
    evaluate.evaluate_cm()
    evaluate.Evaluate_accuracy()
    printErrorIndex(y_predNB, y)
    print()



    # test SVM
    print("SVM")
    classifierSVM = Classifier(X_train, y)
    classifierSVM.SVM()
    y_predSVM = classifierSVM.SVM()
    y_predAll.append(y_predSVM)
    # test evaluate data SVM
    evaluate = Evalauate(y_predSVM, y)
    evaluate.evaluate_cm()
    y_predNB = evaluate.Evaluate_accuracy()
    printErrorIndex(y_predSVM, y)
    print()

    # test Logistik Regression
    print("Logistik Regression")
    classifierLR = Classifier(X_train, y)
    classifierLR.LogistikRegression()
    y_predLR = classifierLR.LogistikRegression()
    y_predAll.append(y_predLR)
    # test evaluate data Logistik Regression
    evaluate = Evalauate(y_predLR, y)
    evaluate.evaluate_cm()
    y_predLR = evaluate.Evaluate_accuracy()
    print()

    # test Random Forest
    print("Random Forest")
    classifierRF = Classifier(X_train, y)
    classifierRF.RandomForest()
    y_predRF = classifierRF.RandomForest()
    y_predAll.append(y_predRF)
    # test evaluate data Random Forest
    evaluate = Evalauate(y_predRF, y)
    evaluate.evaluate_cm()
    y_predRF = evaluate.Evaluate_accuracy()
    print()

    # test Neural Network
    print("Neural Network")
    classifierNN = Classifier(X_train, y)
    classifierNN.NeuralNetwork()
    y_predMLP = classifierNN.NeuralNetwork()
    y_predAll.append(y_predMLP)
    # test evaluate data Neural Network
    evaluate = Evalauate(y_predMLP, y)
    evaluate.evaluate_cm()
    y_predNN = evaluate.Evaluate_accuracy()
    print()

    # test KNN
    print("KNN")
    classifierKNN = Classifier(X_train, y)
    classifierKNN.KNN()
    y_predKNN = classifierKNN.KNN()
    y_predAll.append(y_predKNN)
    # test evaluate data KNN
    evaluate = Evalauate(y_predKNN, y)
    evaluate.evaluate_cm()
    y_predKNN = evaluate.Evaluate_accuracy()
    print()

    # test Decision Tree
    print("Decision Tree")
    classifierDT = Classifier(X_train, y)
    classifierDT.DecisionTree()
    y_predDT = classifierDT.DecisionTree()
    y_predAll.append(y_predDT)
    # test evaluate data Decision Tree
    evaluate = Evalauate(y_predDT, y)
    evaluate.evaluate_cm()
    y_predDT = evaluate.Evaluate_accuracy()
    print()

    # test GaussianNB
    print("GaussianNB")
    classifierGNB = Classifier(X_train, y)
    classifierGNB.GaussianNB()
    y_predGNB = classifierGNB.GaussianNB()
    y_predAll.append(y_predGNB)
    # test evaluate data GaussianNB
    evaluate = Evalauate(y_predGNB, y)
    evaluate.evaluate_cm()
    y_predGNB = evaluate.Evaluate_accuracy()
    print()

    # test GradientBoostingClassifier
    print("GradientBoostingClassifier")
    classifierGB = Classifier(X_train, y)
    classifierGB.GradientBoosting()
    y_predGB = classifierGB.GradientBoosting()
    y_predAll.append(y_predGB)
    # test evaluate data GradientBoostingClassifier
    evaluate = Evalauate(y_predGB, y)
    evaluate.evaluate_cm()
    y_predGB = evaluate.Evaluate_accuracy()
    print()

    #print y_predAll
    print(len(y_predAll))
    print(len(y_predAll[0]))

    # test myEnsamble
    print("myEnsamble")
    classifierME = Classifier(X_train, y)
    y_predME = classifierME.myEnsamble(y_predAll)
    # test evaluate data myEnsamble
    evaluate = Evalauate(y_predME, y)
    evaluate.evaluate_cm()
    y_predME = evaluate.Evaluate_accuracy()


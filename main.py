import nltk
import pandas as pd
from pandas import read_excel
from sklearn.model_selection import train_test_split

import Evaluate
from Evaluate import Evalauate
from Classifier import Classifier
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def printErrorIndex(y_pred, y_test):
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            print(i, y_pred[i], y_test[i])

def evaluateTest(y_pred, y_test):
    result = []
    evaluate = Evaluate.Evalauate(y_pred, y_test)
    result.append(evaluate.Evaluate_accuracy())
    result.append(evaluate.Evaluate_precision())
    result.append(evaluate.Evaluate_precision())
    result.append(evaluate.Evaluate_f1())
    return result
def readData():
    # read from excel
    df = read_excel('Book1.xlsx', sheet_name='Book1')
    # read each column
    X = df['text'].tolist()
    y = df['class'].tolist()
    return X, y

def plotGrafikFromExcel():
    # read from excel
    df = read_excel('hasilpaperrekayasa.xlsx', sheet_name='Sheet1')
    # read each column
    acc = df['Accuracy'].tolist()
    prec = df['Precision'].tolist()
    recall = df['Recall'].tolist()
    f1 = df['F1'].tolist()
    list_method = df['Method'].tolist()

    import matplotlib.pyplot as plt
    # plot bar accuracy
    plt.bar(list_method, acc)
    plt.ylim(75, 100)
    plt.title("Accuracy")
    plt.show()

if __name__ == '__main__':
    plotGrafikFromExcel()
    # X,y = readData()
    #
    # # divide into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # # print length of train and test
    # # print(len(X_train), len(y_train))
    # # print(len(X_test), len(y_test))
    # X = X_train + X_test
    # y = y_train + y_test
    # # print("YYY ", y)
    #
    # # preprocesing text data for Indonesian with sastrawi
    # # create stemmer
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    #
    # # stemming process with sastrawi
    # # for i in range(len(X)):
    # #     X[i] = stemmer.stem(X[i])
    # #
    # # # case folding process sastrawi
    # # for i in range(len(X)):
    # #     X[i] = X[i].lower()
    # #
    # # # stemming proses with sastrawi
    # # for i in range(len(X)):
    # #     X[i] = stemmer.stem(X[i])
    #
    #
    # # test BoW
    # from BoW import boW
    # bow_train = boW(X)
    # X_train = bow_train.vectorizer()
    #
    # #varible for save y_predAll
    # y_predAll = []
    # evaluteAll = []
    #
    # # test classifier Naive Bayes
    # print("Naive Bayes")
    # classifierNB = Classifier(X_train, y)
    # classifierNB.MultinomialBN()
    # y_predNB = classifierNB.MultinomialBN()
    # y_predAll.append(y_predNB)
    # # test evaluate data Naive Bayes
    # evaluteAll.append(evaluateTest(y_predNB, y))
    #
    #
    # # printErrorIndex(y_predNB, y)
    # print()
    #
    #
    #
    # # test SVM
    # print("SVM")
    # classifierSVM = Classifier(X_train, y)
    # classifierSVM.SVM()
    # y_predSVM = classifierSVM.SVM()
    # y_predAll.append(y_predSVM)
    # # test evaluate data SVM
    # evaluate = Evalauate(y_predSVM, y)
    # evaluate.evaluate_cm()
    # y_predNB = evaluate.Evaluate_accuracy()
    # evaluteAll.append(evaluateTest(y_predSVM, y))
    #
    # print()
    #
    # # test Logistik Regression
    # print("Logistik Regression")
    # classifierLR = Classifier(X_train, y)
    # classifierLR.LogistikRegression()
    # y_predLR = classifierLR.LogistikRegression()
    # y_predAll.append(y_predLR)
    # evaluteAll.append(evaluateTest(y_predLR, y))
    # print()
    #
    # # test Random Forest
    # print("Random Forest")
    # classifierRF = Classifier(X_train, y)
    # classifierRF.RandomForest()
    # y_predRF = classifierRF.RandomForest()
    # y_predAll.append(y_predRF)
    # # test evaluate data Random Forest
    # evaluteAll.append(evaluateTest(y_predRF, y))
    # print()
    #
    # # test Neural Network
    # print("Neural Network")
    # classifierNN = Classifier(X_train, y)
    # classifierNN.NeuralNetwork()
    # y_predMLP = classifierNN.NeuralNetwork()
    # y_predAll.append(y_predMLP)
    # # test evaluate data Neural Network
    # evaluteAll.append(evaluateTest(y_predMLP, y))
    # print()
    #
    # # test KNN
    # print("KNN")
    # classifierKNN = Classifier(X_train, y)
    # classifierKNN.KNN()
    # y_predKNN = classifierKNN.KNN()
    # y_predAll.append(y_predKNN)
    # # test evaluate data KNN
    # evaluteAll.append(evaluateTest(y_predKNN, y))
    # print()
    #
    # # test Decision Tree
    # print("Decision Tree")
    # classifierDT = Classifier(X_train, y)
    # classifierDT.DecisionTree()
    # y_predDT = classifierDT.DecisionTree()
    # y_predAll.append(y_predDT)
    # # test evaluate data Decision Tree
    # evaluteAll.append(evaluateTest(y_predDT, y))
    # print()
    #
    # # test GaussianNB
    # print("GaussianNB")
    # classifierGNB = Classifier(X_train, y)
    # classifierGNB.GaussianNB()
    # y_predGNB = classifierGNB.GaussianNB()
    # y_predAll.append(y_predGNB)
    # # test evaluate data GaussianNB
    # evaluteAll.append(evaluateTest(y_predGNB, y))
    # print()
    #
    # # test GradientBoostingClassifier
    # print("GradientBoostingClassifier")
    # classifierGB = Classifier(X_train, y)
    # classifierGB.GradientBoosting()
    # y_predGB = classifierGB.GradientBoosting()
    # y_predAll.append(y_predGB)
    # # test evaluate data GradientBoostingClassifier
    # evaluteAll.append(evaluateTest(y_predGB, y))
    # print()
    #
    # #print y_predAll
    # print(len(y_predAll))
    # print(len(y_predAll[0]))
    #
    # # test myEnsamble
    # print("myEnsamble")
    # classifierME = Classifier(X_train, y)
    # y_predME = classifierME.myEnsamble(y_predAll)
    # # test evaluate data myEnsamble
    # evaluteAll.append(evaluateTest(y_predME, y))
    #
    # # Save EvaluateALL to excel
    # df = pd.DataFrame(evaluteAll)
    # df.to_excel('EvaluateALL.xlsx', index=False, header=False)
    # print("DONE")


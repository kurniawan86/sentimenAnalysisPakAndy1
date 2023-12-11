import pandas as pd
from sklearn.model_selection import train_test_split
from Evaluate import Evalauate
from Classifier import Classifier

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

    # test classifier Naive Bayes
    print("Naive Bayes")
    classifierNB = Classifier(X_train, y)
    classifierNB.MultinomialBN()
    y_predNB = classifierNB.MultinomialBN()
    # test evaluate data Naive Bayes
    evaluate = Evalauate(y_predNB, y)
    evaluate.evaluate_cm()
    evaluate.Evaluate_accuracy()
    print()

    # test SVM
    print("SVM")
    classifierSVM = Classifier(X_train, y)
    classifierSVM.SVM()
    y_pred = classifierSVM.SVM()
    # test evaluate data SVM
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predNB = evaluate.Evaluate_accuracy()
    print()

    # test Logistik Regression
    print("Logistik Regression")
    classifierLR = Classifier(X_train, y)
    classifierLR.LogistikRegression()
    y_pred = classifierLR.LogistikRegression()
    # test evaluate data Logistik Regression
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predLR = evaluate.Evaluate_accuracy()
    print()

    # test Random Forest
    print("Random Forest")
    classifierRF = Classifier(X_train, y)
    classifierRF.RandomForest()
    y_pred = classifierRF.RandomForest()
    # test evaluate data Random Forest
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predRF = evaluate.Evaluate_accuracy()
    print()

    # test Neural Network
    print("Neural Network")
    classifierNN = Classifier(X_train, y)
    classifierNN.NeuralNetwork()
    y_pred = classifierNN.NeuralNetwork()
    # test evaluate data Neural Network
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predNN = evaluate.Evaluate_accuracy()
    print()

    # test KNN
    print("KNN")
    classifierKNN = Classifier(X_train, y)
    classifierKNN.KNN()
    y_pred = classifierKNN.KNN()
    # test evaluate data KNN
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predKNN = evaluate.Evaluate_accuracy()
    print()

    # test Decision Tree
    print("Decision Tree")
    classifierDT = Classifier(X_train, y)
    classifierDT.DecisionTree()
    y_pred = classifierDT.DecisionTree()
    # test evaluate data Decision Tree
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predDT = evaluate.Evaluate_accuracy()
    print()

    # test GaussianNB
    print("GaussianNB")
    classifierGNB = Classifier(X_train, y)
    classifierGNB.GaussianNB()
    y_pred = classifierGNB.GaussianNB()
    # test evaluate data GaussianNB
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predGNB = evaluate.Evaluate_accuracy()
    print()

    # test GradientBoostingClassifier
    print("GradientBoostingClassifier")
    classifierGB = Classifier(X_train, y)
    classifierGB.GradientBoosting()
    y_pred = classifierGB.GradientBoosting()
    # test evaluate data GradientBoostingClassifier
    evaluate = Evalauate(y_pred, y)
    evaluate.evaluate_cm()
    y_predGB = evaluate.Evaluate_accuracy()
    print()


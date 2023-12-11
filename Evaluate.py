class Evalauate:
    y_pred = []
    y_test = []

    def __init__(self, Y_pred, Y_test):
        self.y_pred = Y_pred
        self.y_test = Y_test

    def evaluate_cm(self):
        # import library
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        # confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Convution Matrix :",cm)
        # accuracy score
        # accuracy = accuracy_score(self.y_test, self.y_pred)
        # print("Accuracy :", accuracy)
        # # precision score
        # from sklearn.metrics import precision_score
        # precision = precision_score(self.y_test, self.y_pred, average='macro')
        # print("Precision :",precision)
        # # recall score
        # from sklearn.metrics import recall_score
        # recall = recall_score(self.y_test, self.y_pred, average='macro')
        # print("Recall :", recall)
        # # f1 score
        # from sklearn.metrics import f1_score
        # f1 = f1_score(self.y_test, self.y_pred, average='macro')
        # print("f1 score :",f1)
        # # classification report
        # from sklearn.metrics import classification_report
        # print(classification_report(self.y_test, self.y_pred))
        # # ROC AUC score
        # from sklearn.metrics import roc_auc_score
        # roc_auc = roc_auc_score(self.y_test, self.y_pred)
        # print("roc_auc", roc_auc)
        # # ROC curve
        # from sklearn.metrics import roc_curve
        # import matplotlib.pyplot as plt
        # fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        # plt.plot(fpr, tpr)
        # plt.title("roc_curve")
        # plt.show()
        # # precision recall curve
        # from sklearn.metrics import precision_recall_curve
        # import matplotlib.pyplot as plt
        # precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred)
        # plt.plot(recall, precision)
        # plt.title("Precesion Recall")
        # plt.show()

    def Evaluate_accuracy(self):
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy :", accuracy)
        return accuracy

    def Evaluate_precision(self):
        from sklearn.metrics import precision_score
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        print("Precision :",precision)
        return precision

    def Evaluate_recall(self):
        from sklearn.metrics import recall_score
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        print("Recall :", recall)
        return recall

    def Evaluate_f1(self):
        from sklearn.metrics import f1_score
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        print("f1 score :",f1)
        return f1

    def Evaluate_roc_auc(self):
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(self.y_test, self.y_pred)
        print("roc_auc", roc_auc)
        return roc_auc

    def Evaluate_roc_curve(self):
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        plt.plot(fpr, tpr)
        plt.title("roc_curve")
        plt.show()

    def Evaluate_precision_recall_curve(self):
        from sklearn.metrics import precision_recall_curve
        import matplotlib.pyplot as plt
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred)
        plt.plot(recall, precision)
        plt.title("Precesion Recall")
        plt.show()


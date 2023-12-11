class Classifier():
    x_train = []
    y_train = []
    classifier = None

    def __init__(self, X_train, Y_train):
        self.x_train = X_train
        self.y_train = Y_train


    def MultinomialBN(self) :
        # import library
        from sklearn.naive_bayes import MultinomialNB
        # create object
        classifier = MultinomialNB()
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred


    def SVM(self):
        # import library
        from sklearn.svm import SVC
        # create object
        classifier = SVC(kernel='linear')
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def LogistikRegression(self):
        # import library
        from sklearn.linear_model import LogisticRegression
        # create object
        classifier = LogisticRegression()
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def RandomForest(self):
        # import library
        from sklearn.ensemble import RandomForestClassifier
        # create object
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def NeuralNetwork(self):
        # import library
        from sklearn.neural_network import MLPClassifier
        # create object
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def KNN(self):
        # import library
        from sklearn.neighbors import KNeighborsClassifier
        # create object
        classifier = KNeighborsClassifier(n_neighbors=3)
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def DecisionTree(self):
        # import library
        from sklearn import tree
        # create object
        classifier = tree.DecisionTreeClassifier()
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def GaussianNB(self):
        # import library
        from sklearn.naive_bayes import GaussianNB
        # create object
        classifier = GaussianNB()
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred

    def GradientBoosting(self):
        # import library
        from sklearn.ensemble import GradientBoostingClassifier
        # create object
        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        # train model
        classifier.fit(self.x_train, self.y_train)
        self.classifier = classifier
        y_pred = self.classifier.predict(self.x_train)
        return y_pred



class StandardClassifier_1D:

    def __init__(self, model_name, **model_params):
        self.model_name = model_name
        self.model = None
        if self.model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(**model_params)
        elif self.model_name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**model_params)
        elif self.model_name == 'LinearSVC':
            from sklearn.svm import LinearSVC
            self.model = LinearSVC(**model_params)
        elif self.model_name == 'RBF SVC':
            from sklearn.svm import SVC
            self.model = SVC(**model_params)
        elif self.model_name == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(**model_params)
        elif self.model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**model_params)
        elif self.model_name == 'GradientBoost':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(**model_params)
        elif self.model_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier
            self.model = AdaBoostClassifier(**model_params)
        elif self.model_name == 'XGBoost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**model_params)

    def fit(self, trainX, trainY, validX, validY):
        trainX = trainX.reshape((-1, 258))  # TODO: A hack for now
        print(self.model)
        self.model.fit(trainX, trainY.ravel())

    def predict(self, testX):
        testX = testX.reshape((-1, 258))  # TODO: A hack for now
        return self.model.predict(testX)

    def save(self, path):
        # save the model to disk
        import pickle
        filename = path + self.model_name + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        # Test
        # load the model from disk
        import pickle
        filename = path + self.model_name + '.sav'
        self.model = pickle.load(open(filename, 'rb'))
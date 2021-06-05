
class StandardRegressor_1D:

    def __init__(self, model_name, **model_params):
        self.model_name = model_name
        self.model = None
        if self.model_name == 'KNN':
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(model_params)
        elif self.model_name == 'LinearReg':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(model_params)
        elif self.model_name == 'Ridge':
            from sklearn.linear_model import Ridge
            self.model = Ridge(model_params)
        elif self.model_name == 'Lasso':
            from sklearn.linear_model import Lasso
            self.model = Lasso(model_params)
        elif self.model_name == 'ElasticNet':
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(model_params)
        elif self.model_name == 'RBF SVR':
            from sklearn.svm import SVR
            self.model = SVR(model_params)
        elif self.model_name == 'DecisionTree':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(model_params)
        elif self.model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(model_params)
        elif self.model_name == 'GradientBoost':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(model_params)
        elif self.model_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostRegressor
            self.model = AdaBoostRegressor(model_params)
        elif self.model_name == 'XGBoost':
            from xgboost import XGBRegressor
            self.model = XGBRegressor(model_params)

    def fit(self, trainX, trainY, validX, validY):
        trainX = trainX.reshape((-1, 258))  # TODO: A hack for now
        self.model.fit(trainX, trainY.ravel())

    def predict(self, testX):
        testX = testX.reshape((-1, 258))  # TODO: A hack for now
        return self.model.predict(testX)
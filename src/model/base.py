from sklearn.model_selection import GridSearchCV


class BaseModel(object):

    def __init__(self):
        self.model = None
        self.tuned_parameters = None

    def train(self, data, target, cv=True):

        if cv:
            self.model = GridSearchCV(
                self.model,
                self.tuned_parameters,
                cv=5)
            self.model.fit(data, target)
            print(self.model.best_params_)
        else:
            self.model.fit(data, target)

    def predict(self, data):
        return self.model.predict(data)

    def dump():
        pass

    def load():
        pass

    def decision_function(self, y_test):
        return self.model.predict_proba(y_test)

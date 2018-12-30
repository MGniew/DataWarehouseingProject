from sklearn.neural_network import MLPClassifier

from src.model.base import BaseModel


class MLPModel(BaseModel):

    def __init__(self):
        super(MLPModel, self).__init__()
        self.model = MLPClassifier(activation="logistic")
        self.tuned_parameters = {
            "activation": ["logistic", "tanh", "relu"]}

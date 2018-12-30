from sklearn.svm import SVC

from src.model.base import BaseModel


class SVMModel(BaseModel):

    def __init__(self):
        super(SVMModel, self).__init__()
        self.model = SVC(probability=True, gamma="auto", kernel="rbf", C=100)
        self.tuned_parameters = [
            {'kernel': ['linear', "rbf"], 'C': [1, 10, 100, 1000]}]

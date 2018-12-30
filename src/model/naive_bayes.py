from sklearn.naive_bayes import MultinomialNB

from src.model.base import BaseModel


class NaiveBayesModel(BaseModel):

    def __init__(self):
        super(NaiveBayesModel, self).__init__()
        self.model = MultinomialNB(alpha=0.01)
        self.tuned_parameters = {"alpha": [0.01, 0.1, 0.25, 0.5, 0.75, 1]}

from src.model.base import BaseModel

from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel(BaseModel):

    def __init__(self):
        super(DecisionTreeModel, self).__init__()
        self.model = DecisionTreeClassifier(min_samples_split=2)
        self.tuned_parameters = {"min_samples_split": range(2, 400, 10)}

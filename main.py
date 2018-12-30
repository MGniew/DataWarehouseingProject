import argparse

from src.data_preparation.data_loader import load_df_from_csv
from src.data_preparation.data_loader import split_data
from src.data_preparation.data_loader import get_oversampled_data
from src.data_preparation.reduction import get_selected_features
from src.data_preparation.reduction import get_reduced_dimension

from src.model.naive_bayes import NaiveBayesModel
from src.model.mlp import MLPModel
from src.model.decision_tree import DecisionTreeModel
from src.model.svm import SVMModel

from src.score.scorer import get_auc
from src.score.scorer import get_roc_curve
from src.score.scorer import get_confusion_matrix
from src.score.scorer import get_classification_report

from src.visualisation.plot_maker import curve_plot, heatmap, scatter_plot

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    data = load_df_from_csv(args.input_file)
    #  heatmap(data)

    data = get_oversampled_data(data, ratio=1)

    target = data.pop("target")
    data = get_selected_features(data, target, "vt")

    #cor = get_reduced_dimension(data)
    #scatter_plot(cor[:, 0], cor[:, 1], target)

    data, test = split_data(data, [0.8])
    target, test_target = split_data(target, [0.8])

    model = SVMModel()
    model.train(data, target, True)
    y_pred = model.predict(test)

    print(get_classification_report(test_target, y_pred))
    print(get_confusion_matrix(test_target, y_pred))

    y_pred = model.decision_function(test)[:, 1]

    fpr, tpr = get_roc_curve(test_target, y_pred)
    auc = get_auc(fpr, tpr)
    curve_plot(fpr, tpr, auc)


if __name__ == "__main__":
    main()

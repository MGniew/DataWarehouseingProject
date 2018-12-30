import pandas as pd


def load_df_from_csv(filename):
    data = pd.read_csv(filename)
    data['target'] = data['target'].map({'yes': 1, 'no': 0})
    return data


def split_data(data, sizes=[0.6, 0.8]):

    if not all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)):
        raise ValueError

    if not all(0 < s < 1 for s in sizes):
        raise ValueError

    sizes.insert(0, 0)
    sizes.append(1)
    sizes = [int(el * len(data)) for el in sizes]

    result = []

    while len(sizes) > 1:
        result.append(data[sizes[0]:sizes[1]])
        sizes = sizes[1:]

    return result


def get_oversampled_data(data, label=1, ratio=2):

    if ratio < 0:
        ratio = 1

    data_label = data.loc[data['target'] == label]
    data = data.append(data_label.sample(frac=ratio, replace=True))

    return data.sample(frac=1)

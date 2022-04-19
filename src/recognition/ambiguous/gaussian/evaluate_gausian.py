import pandas as pd

from sklearn.metrics import accuracy_score

from recognition.ambiguous.gaussian import GaussianRecognition


def cluster_to_str(clusters):
    array = [len(c) for c in set(clusters)]
    return str(array)


if __name__ == '__main__':
    test = pd.read_csv('../../../../data/23_schemes/test.csv')  # TODO: series data to evaluate

    X = test.values[:100, :-1].astype(float)
    y = test.scheme[:100]

    y_pred = []
    for i, (x, real) in enumerate(zip(X, y)):
        rec = GaussianRecognition()
        rec.fit(x)
        pred = cluster_to_str(rec.big_lines_)
        y_pred.append(pred)
        print(i, real, pred)

    print(accuracy_score(y, y_pred))

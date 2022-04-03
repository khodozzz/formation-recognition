import pandas as pd

from sklearn.metrics import accuracy_score
from recognition.unambiguous.clustering.forel import FOREL


def cluster_to_str(clusters):
    array = [len(c) for c in clusters]
    return str(array)


if __name__ == '__main__':
    test = pd.read_csv('../../../../data/test.csv')

    X = test.values[:, :-1].astype(float)
    y = test.scheme

    clustering = FOREL()

    y_pred = []
    for x in X:
        clustering.fit(x)
        y_pred.append(cluster_to_str(clustering.clusters_))

    print(accuracy_score(y, y_pred))

    for i, (real, pred) in enumerate(zip(y, y_pred)):
        print(i, real, pred)


import pandas as pd

from sklearn.cluster import (AgglomerativeClustering, DBSCAN, Birch)
from sklearn.metrics import accuracy_score

from recognition.ambiguous.multiply_clustering import MultiplyClusteringChangingParam


def cluster_to_str(clusters):
    array = [clusters.count(c) for c in set(clusters)]
    return str(array)


if __name__ == '__main__':
    test = pd.read_csv('../../../../data/test.csv')

    X = test.values[:100, :-1].astype(float)
    y = test.scheme[:100]

    method_param = {DBSCAN(min_samples=1): 'eps',
                    AgglomerativeClustering(n_clusters=None): 'distance_threshold',
                    Birch(n_clusters=None): 'threshold'}

    for method in method_param:
        m_name = str(method).split('(')[0]
        print(m_name)

        y_pred = []
        for i, (x, real) in enumerate(zip(X, y)):
            clus = MultiplyClusteringChangingParam(method, method_param[method])
            clus.fit(x, verbose=False, plot_heatmap=False)
            pred = cluster_to_str(clus.labels_)
            y_pred.append(pred)
            print(i, real, pred)

        print(accuracy_score(y, y_pred))

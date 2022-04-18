import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import (AgglomerativeClustering, DBSCAN, Birch)


class MultiplyClusteringChangingParam:
    def __init__(self, method,
                 param_name, param_start=0, param_step=0.01,
                 threshold=0.9):
        self.method = method
        self.param_name = param_name
        self.param_start = param_start
        self.param_step = param_step
        self.threshold = threshold

        self.Y = None
        self.labels_ = None

    def fit(self, X, verbose=True):
        self._calc_Y(X, verbose)
        self._calc_heatmap()
        self._calc_labels()

    def _calc_Y(self, X, verbose):
        X_sorted = np.array(sorted(X))[:, np.newaxis]
        self.Y = []

        param = self.param_start
        clust_prev = None
        while clust_prev is None or max(clust_prev) != 0:  # change stop condition?
            param += self.param_step

            self.method.set_params(**{self.param_name: param})
            clust = self.method.fit(X_sorted).labels_

            if (clust_prev != clust).any():  # output when labels changed
                clust_prev = clust
                if verbose:
                    print(f"{param:.{2}f}" ':', clust)

            if max(clust) != 9 and max(clust) != 0:  # add all label vectors exсept [0 ... 9] and [0 ... 0]
                self.Y.append(clust)

    def _calc_heatmap(self):
        arr_size = len(self.Y[0])

        self.hmap_ = np.ones((arr_size, arr_size))
        for p1 in range(arr_size):
            for p2 in range(p1, arr_size):
                if p1 == p2:
                    continue

                k = 0
                for i in range(len(self.Y)):
                    if self.Y[i][p1] == self.Y[i][p2]:
                        k += 1

                self.hmap_[p1][p2] = self.hmap_[p2][p1] = k / len(self.Y)

    def _calc_labels(self):
        arr_size = len(self.Y[0])
        self.labels_ = [-1 for _ in range(arr_size)]
        counter = 0

        for i in range(arr_size):
            if self.labels_[i] != -1:  # already labeled
                continue

            self.labels_[i] = counter
            for j in range(i, arr_size):
                if self.labels_[j] == -1 and self.hmap_[i][j] > self.threshold:
                    self.labels_[j] = counter

            counter += 1


def plot_heatmap(hmap, plot_name):
    sns.set_theme()
    sns.heatmap(hmap, cmap=sns.color_palette("Blues", as_cmap=True))
    plt.title(plot_name)
    plt.show()


if __name__ == '__main__':
    my_test = [15, 16, 17, 33, 35, 43, 44, 51, 55, 56]

    method_param = {DBSCAN(min_samples=1): 'eps',
                    AgglomerativeClustering(n_clusters=None): 'distance_threshold',
                    Birch(n_clusters=None): 'threshold'}

    for method in method_param:
        m_name = str(method).split('(')[0]

        clus = MultiplyClusteringChangingParam(method, method_param[method])
        clus.fit(my_test)

        plot_heatmap(clus.hmap_, m_name)

        print(m_name)
        print('#', clus.labels_, '#')

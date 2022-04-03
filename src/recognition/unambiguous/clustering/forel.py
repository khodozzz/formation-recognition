class FOREL:
    def __init__(self, center_func=lambda c: sum(c) / len(c)):
        self.center_func = center_func

        self.clusters_ = None

    def fit(self, X):
        self.__init_components__(X)

        while not self.__recognition_finished__():
            current_value = self.__next_component_value__()
            cluster = self.__form_cluster__(current_value)
            center = self.__calc_center__(cluster)

            while abs(current_value - center) > 0.01:
                current_value = center
                cluster = self.__form_cluster__(current_value)
                center = self.__calc_center__(cluster)

            self.__save_cluster__(cluster)

    def __init_components__(self, X):
        self.clusters_ = []

        self._not_in_line = X.copy()

        self.expected_lines_count_ = 3 / 10 * len(X)
        self.radius_ = (max(X) - min(X)) / (2 * self.expected_lines_count_)

    def __recognition_finished__(self):
        return len(self._not_in_line) == 0

    def __next_component_value__(self):
        return min(self._not_in_line)

    def __form_cluster__(self, center):
        return [p for p in self._not_in_line if abs(p - center) < self.radius_]

    def __calc_center__(self, cluster):
        return self.center_func(cluster)

    def __save_cluster__(self, cluster):
        for p in cluster:
            if p in self._not_in_line:
                self._not_in_line = self._not_in_line[self._not_in_line != p]

        self.clusters_.append(cluster)


if __name__ == '__main__':
    forel = FOREL()
    my_test = [5, 6, 7, 33, 35, 33, 37, 30, 99, 108]
    forel.fit(my_test)
    print(forel.clusters_)

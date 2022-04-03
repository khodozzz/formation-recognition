from data_generation.create_series_dataset import *
from scipy.integrate import *
from scipy import stats
from itertools import *


class GaussianRecognition:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

        self.X = None
        self.x_space_ = None
        self.distributions_ = None
        self.probs_ = None
        self.lines_ = None
        self.big_lines_ = None

    def fit(self, X, plot=False):
        self.X = X
        self._calc_x_space()
        self._calc_distributions(plot)
        self._calc_lines_probs()
        self._define_lines()
        self._define_big_lines()

    def _calc_x_space(self):
        min_x = min([min(x) for x in self.X])
        max_x = max([max(x) for x in self.X])
        self.x_space_ = np.linspace(min_x, max_x)

    def _calc_distributions(self, plot=True):
        self.distributions_ = []
        for x_coords in self.X:
            y = stats.norm.pdf(self.x_space_, np.mean(x_coords), np.std(x_coords))
            self.distributions_.append(y)
            if plot:
                plt.plot(self.x_space_, y)
        if plot:
            plt.show()

    def _calc_lines_probs(self):
        all_combinations = list(chain(*map(lambda x: combinations(range(len(self.distributions_)), x),
                                           range(2, len(self.distributions_) + 1))))

        self.probs_ = {}
        for comb in all_combinations:
            union = [self.distributions_[i] for i in comb]
            min_union = union[0].copy()
            for u in union:
                for k in range(len(min_union)):
                    if u[k] < min_union[k]:
                        min_union[k] = u[k]
            self.probs_[comb] = simpson(x=self.x_space_, y=min_union)

    def _define_lines(self):
        self.lines_ = []
        for comb in self.probs_:
            if self.probs_[comb] > self.threshold:
                self.lines_.append(comb)

    def _define_big_lines(self):
        self.big_lines_ = self.lines_.copy()

        remove_list = []
        for smaller in self.lines_:
            for bigger in self.lines_:
                if smaller is not bigger:
                    is_remove = True
                    for el in smaller:
                        if el not in bigger:
                            is_remove = False
                            break
                    if is_remove:
                        remove_list.append(smaller)
                        break

        for line in remove_list:
            self.big_lines_.remove(line)


if __name__ == '__main__':
    players_num = 10

    sch = gen_scheme([3, 4, 3], std_divider=4, series_time=15)

    rec = GaussianRecognition(threshold=0.5)
    rec.fit(sch, plot=True)

    for line in rec.big_lines_:
        print(f'{line} : {rec.probs_[line]}')
        #
        # ax = plt.gca()
        # ax.set_ylim([0, max([max(d_) for d_ in d])])
        #
        # plt.plot(x, min_union)
        # plt.title(f'{comb}')
        # plt.show()

from sklearn import datasets
from itertools import product
from collections import OrderedDict
from sklearn.cluster import KMeans


def get_data():
    n_samples = 2000
    n_features = 2
    n_centers = 3
    X, _ = datasets.samples_generator.make_blobs(n_samples=n_samples,
                                                 n_features=n_features,
                                                 centers=n_centers,
                                                 random_state=None)
    return X


def convert_dict_to_ordered(dict_to_convert):
    new_dict = OrderedDict()
    for key, value in dict_to_convert.iteritems():
        new_dict[key] = value
    return new_dict


def convert_product_to_param_dict(param_tuple):
    param_dict = {
        'n_init': param_tuple[0],
        'n_clusters': param_tuple[1]
        }
    return param_dict


class clustering_pipeline(object):

    def __init__(self, X, param_dict):
        self.scoring_log = []
        self.X = X
        self.param_dict = param_dict
        # self.evaluate_silhouette()
        self.grid_search()

    def fit(self, X, param_dict):
        km = KMeans(n_clusters=param_dict.get('n_clusters'),
                    init='random',
                    n_init=param_dict.get('n_init'),
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        km.fit_predict(X)
        self.evaluate_silhouette(km, param_dict)

    def grid_search(self):
        param_dict = convert_dict_to_ordered(self.param_dict)
        for perm in product(*param_dict.itervalues()):
            self.fit(self.X, convert_product_to_param_dict(perm))

    def evaluate_silhouette(self, km, param_dict):
        self.log_params(km.inertia_, param_dict)

    def log_params(self, inertia, param_dict):
        d = {'param_dict': param_dict, 'inertia': inertia}
        self.scoring_log.append(d)


def main():
    X = get_data()
    param_dict = {
            'n_clusters': [2, 3, 4, 5, 6],
            'n_init': [10, 15, 20, 25, 30]
            }
    cm = clustering_pipeline(X, param_dict)  # .foo()
    print(cm.scoring_log)


if __name__ == '__main__':
    main()

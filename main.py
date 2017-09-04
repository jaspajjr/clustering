from sklearn import datasets
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


class clustering_pipeline(object):

    def __init__(self, X, param_dict):
        self.scoring_log = []
        self.X = X
        self.param_dict = param_dict
        self.fit()
        self.evaluate_silhouette()

    def get_params(self):
        n_clusters = self.param_dict.get('n_clusters', 3)
        return n_clusters

    def fit(self):
        n_clusters = self.get_params()
        self.km = KMeans(n_clusters=n_clusters,
                         init='random',
                         n_init=10,
                         max_iter=300,
                         tol=1e-04,
                         random_state=0)
        self.km.fit_predict(self.X)

    def evaluate_silhouette(self):
        self.log_params(self.km.inertia_)
        print(self.km.inertia_)

    def log_params(self, inertia):
        d = {'param_dict': self.param_dict, 'inertia': inertia}
        self.scoring_log.append((self.param_dict, d))


def main():
    X = get_data()
    param_dict = {'n_clusters': 2}
    cm = clustering_pipeline(X, param_dict)  # .foo()
    print(cm.scoring_log)


if __name__ == '__main__':
    main()

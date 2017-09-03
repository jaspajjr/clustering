from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


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
        print(self.km.inertia_)


def main():
    X = get_data()
    param_dict = {'n_clusters': 2}
    cm = clustering_pipeline(X, param_dict)  # .foo()
    print(type(cm.km))


if __name__ == '__main__':
    main()

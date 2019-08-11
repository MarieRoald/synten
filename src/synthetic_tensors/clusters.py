from sklearn.datasets import make_classification

class ClusterFactors:
    def __init__(self, num_samples, num_components):
        self.num_samples = num_samples
        self.num_components = num_components

    def init_clusters(self, num_informative_components=None, class_sep=2):

        if num_informative_components is None:
            num_informative_components = self.num_components

        

        A, A_class = make_classification(
            n_samples=self.num_samples,
            n_features=self.num_components,
            n_informative=num_informative_components,
            n_redundant=0,
            n_clusters_per_class = 1,
            class_sep = class_sep
        )

        self.factor_matrix = A
        self.classes = A_class




class ClusterFactorGenerator:
    def __init__(self, generator_kwargs, init_kwargs):
        self.generator = ClusterFactors(**generator_kwargs)
        self.init_kwargs = init_kwargs
    
    def generate_factors(self):
        self.generator.init_clusters(**self.init_kwargs)
        return self.generator.factor_matrix
    
    @property
    def classes(self):
        return self.generator.classes

        
import warnings
from sklearn.datasets import make_classification


class ClusterComponents:
    # Open whole make_classification API
    def __init__(self, num_samples, num_components, num_informative_components=None, class_sep=2, random_state=None):
        if num_informative_components is None:
            num_informative_components = num_components
        
        self.num_samples = num_samples
        self.num_components = num_components
        self.random_state = random_state
        self.num_informative_components = num_informative_components
        self.class_sep=class_sep

    def init_clusters(self,):
        # Remove this function
        A, A_class = make_classification(
            n_samples=self.num_samples,
            n_features=self.num_components,
            n_informative=self.num_informative_components,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=self.class_sep,
            random_state=self.random_state,
        )

        self.factor_matrix = A
        self.classes = A_class


class ClusterGenerator:
    def __init__(self, generator_kwargs, random_state=None):
        warnings.warn('Needs refactoring', RuntimeWarning)
        self.generator = ClusterComponents(random_state=random_state, **generator_kwargs)
    
    def generate_factors(self):
        self.generator.init_clusters()
        return self.generator.factor_matrix
    
    @property
    def classes(self):
        return self.generator.classes

        
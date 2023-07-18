from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    random_state = 42

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        # making random symmetric matrices
        random_matrices = []
        for i in range(576):
            a = np.random.uniform(0, 1)
            d = np.random.uniform(0, 1)
            b = np.random.uniform(-1, 1)
            random_matrix = np.array([[a, b], [0, d]])
            random_matrix = np.dot(random_matrix, random_matrix.transpose())
            random_matrices.append(random_matrix)

        X = np.array(random_matrices)
        y = np.random.randint(0, 8, 576)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)

from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    import pickle
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Cov_mat"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        # the data ued here are the covariances matrices of the dataset

        link = '/Users/christophermarouani/Desktop/TIPE/features_cov.pickle'
        with open(str(link),
                  'rb') as handle:
            out_pickle = pickle.load(handle)

        matrices = out_pickle['data']
        classes = out_pickle['labels_code']
        X = np.array(matrices)
        y = np.array(classes)
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)

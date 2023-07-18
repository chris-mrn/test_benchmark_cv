from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.

with safe_import_context() as import_ctx:
    from sklearn.model_selection import KFold
# et peut être ici définir un import vers la fonction qui
# fera bien la cross_validation

# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "BCI_test"

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    # min_benchopt_version = "1.3" we erase this due to the fact that we have
    # the tag is not up to date ( but the repo is) we have 0.1.dev585
    # go here to solve the issue :
    # https://stackoverflow.com/questions/51006930/upstream-git-tag-is-not-showing-in-forked-repository

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    # cv object from sklearn

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        # if you want to split the data it could be here
        # set the data we want to split

        self.X, self.y = X, y

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.

        return dict(score_test=score_test,
                    value=-score_test,
                    score_train=score_train)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        # if you have define a cv objective, get_split
        # will iterate the split of the data using the cv object

        X_train, X_test, y_train, y_test = self.get_split(self.X, self.y)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        return dict(X=self.X_train, y=self.y_train)

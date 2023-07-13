from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.

with safe_import_context() as import_ctx:
    from sklearn.model_selection import KFold
    import numpy as np
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

    evaluation_process = 'cross_validation'
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    # cv object from sklearn

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        # if you want to split the data it could be here

        self.X, self.y = X, y
        self.cv = self.cv.split(X)
        # ou mettre cette ligne ?
        # pour moi elle ne doit pas être vue par l'utilisateur
        # on doit fixer cv avant de rentrern

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        print("dans compute le train est", self.y_train[:10])
        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.

        return dict(score_test=score_test,
                    value=-score_test,
                    score_train=score_train)
    '''
    en fait on voudrait avoir le split déja fait par sklearn
    qui donne une liste de (X_train,X_test) ou du moins les indices
    nécessaires au split et ensuite d'itérer sur cette liste en faisant des
    # sur chacun des élements qui feront chacun un run du solver
    # ensuite benchopt sortira sur le html le moyenne des socres des runs
    #  es ce que ca itère aussi tout le benchmark?
    # on espère pas parce que sinon à chaque split ca ferait tout le process

    '''

    '''
     en fait on veut le procéder de split soit ittératif de sorte que si je
     rappelle le split ça me sorte un train et test disjoint du précédent
     de la même manière que la cross val crée 5 folds et va crée de
     train et test disjoints et faire la moyenne sur les folds

     '''
    '''
    define the way we are splittingt the data, for example  cross_sessions
    or cross_subject
    avoir une liste avec des données sur lesquelles je peux m'entrainer ou non
    '''

    def split(self, train_index, test_index):
        # Return the splited data
        X = self.X
        y = self.y
        X_train, y_train = [], []
        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])

        X_test, y_test = [], []
        for i in test_index:
            X_test.append(X[i])
            y_test.append(y[i])

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return X_train, X_test, y_train, y_test

    # si on veut récupérer les donnée pour le solver
    # sous une autre forme que array

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        # get_split will definie the split for the cross_val

        X_train, X_test, y_train, y_test = self.get_split()
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        return dict(X=self.X_train, y=self.y_train)

import logging
import multiprocessing

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression

from ._model import DecisionTreeBinaryClassifier

logger = logging.getLogger(__name__)


class BinaryClassifierMixin(ClassifierMixin):
    def _pre_fit(self, X, y):
        assert y.ndim == 1
        ys, nys = np.unique(y, return_counts=True)
        assert ys.tolist() == [0, 1]
        logger.info('fitting %s with X[%d x %d] and y[%d], containing 0[%d] and 1[%d]',
                    str(self), *X.shape, sum(nys), *nys)

    def _post_fit(self, X, y):
        pass

    def predict_proba_one(self, x):
        p = self.predict_proba(np.array(x).reshape(1, -1))
        return p[0, :]


class NaiveBinaryClassifier(BaseEstimator, BinaryClassifierMixin):
    def __init__(self):
        self._params = [1.0, 0.0]

    def fit(self, X, y, sample_weight=None):
        self._pre_fit(X, y)
        # Agresti-Coull formula:
        n1 = y.sum()
        n = len(y)
        r = (n1 + 2.0) / n
        self._params[0] = 1 - r
        self._params[1] = r
        self._post_fit(X, y)
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.repeat([self._params], n, axis=0)

    def predict_proba_one(self, x):
        return self._params.copy()


class BinaryLogisticRegression(LogisticRegression, BinaryClassifierMixin):
    def fit(self, X, y, **kwargs):
        self._pre_fit(X, y)
        LogisticRegression.fit(self, X, y, **kwargs)
        self._post_fit(X, y)
        return self


class RandomForestBinaryClassifier(ForestClassifier, BinaryClassifierMixin):
    """
    A variant of `sklearn.forest.RandomForestClassifier` that uses
    the custom `DecisionTreeBinaryClassifier` as internal workhorse.
    """
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=0,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        if n_jobs == 0:
            n_jobs = multiprocessing.cpu_count()
        ForestClassifier.__init__(
            self,
            base_estimator=DecisionTreeBinaryClassifier(),
                # this `base_estimator` is our customization.
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=None)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split

    def fit(self, X, y, **kwargs):
        self._pre_fit(X, y)
        super().fit(X, y, **kwargs)
        self._post_fit(X, y)
        return self

    def predict_proba_one(self, x):
        """Predict class probabilities for `x`.

        Args:
            x: 1-D `numpy` array of predictors, of type `numpy.float64`.
                This must be an array occupying contiguous memory,
                i.e., it can't be a slice that 'jumps' over elements.
        """
        proba = self.estimators_[0].predict_proba_one(x)
        for t in self.estimators_[1:]:
            p = t.predict_proba_one(x)
            proba += p
        proba /= len(self.estimators_)
        out = np.array([1.0 - proba, proba])
        return out


class GradientBoostingBinaryClassifier(GradientBoostingClassifier, BinaryClassifierMixin):
    def __init__(self, n_estimators=20, learning_rate=0.1):
        GradientBoostingClassifier.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

    def fit(self, X, y, **kwargs):
        self._pre_fit(X, y)
        GradientBoostingClassifier.fit(self, X, y, **kwargs)
        self._post_fit(X, y)
        return self


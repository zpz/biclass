import logging

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree._tree cimport Tree, Node, SIZE_t

logger = logging.getLogger(__name__)


class DecisionTreeBinaryClassifier(DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        assert self.n_outputs_ == 1
        assert self.n_classes_ == 2
        return self

    def predict_proba_one(self, double[:] x):
        """
        A custom version of `DecisionTreeClassifier.predict_proba`.

        `x` is a 1D `numpy.ndarray` with `dtype` of `numpy.float64`.

        `x` is treated read-only in this method.

        Returns probability of class `1`.

        This function is a hack/simplification/customization of
        `sklearn.tree._tree.Tree.predict` version 0.18.1.
        """
        cdef SIZE_t node_idx = _apply_dense(self.tree_, x)
        cdef double[:,:,:] values = self.tree_.value
        cdef double p0, p1, normalizer
        p0 = values[node_idx, 0, 0]
        p1 = values[node_idx, 0, 1]
        normalizer = p0 + p1
        p1 /= normalizer
        return p1


cdef SIZE_t _apply_dense(Tree tree, double[:] x):
    """Finds the terminal region (=leaf node) for X."""

    cdef Node* nodes = <Node *> tree.nodes
    cdef Node* node = nodes

    cdef SIZE_t _TREE_LEAF = -1

    cdef float node_value
    # The tree's node values are saved as `float32`.
    # To ensure reproducible results in corner cases,
    # convert the input from `float64` to `float32`.

    while node.left_child != _TREE_LEAF:
        # # ... and node.right_child != _TREE_LEAF:
        # logger.debug('node.feature: %d; node_value: %f; node.thresh: %f',
        #              node.feature, x[node.feature], node.threshold)
        node_value = x[node.feature]
        #if x[node.feature] <= node.threshold:
        if node_value <= node.threshold:
            node = &nodes[node.left_child]
        else:
            node = &nodes[node.right_child]

    cdef SIZE_t out = <SIZE_t>(node - nodes)  # node offset

    #logger.debug('node idx: %d', out)
    return out

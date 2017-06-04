import logging

from collections import defaultdict

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics.classification import log_loss
from sklearn import model_selection

from .binary_classifier import (
    NaiveBinaryClassifier, BinaryLogisticRegression,
    RandomForestBinaryClassifier, GradientBoostingBinaryClassifier)

logger = logging.getLogger(__name__)


def unbalanced_sample_weight(y):
    w = compute_sample_weight('balanced', y)
    return w


def logloss(act, pred, class_weight=None):
    if class_weight == 'balanced':
        sample_weight = unbalanced_sample_weight(act)
    else:
        sample_weight = None
    return log_loss(act, pred, sample_weight=sample_weight)


def do_model(model, X_train, y_train, X_test, y_test, class_weight=None):
    if class_weight == 'balanced':
        sample_weight = unbalanced_sample_weight(y_train)
    else:
        sample_weight = None
    model.fit(X_train, y_train, sample_weight=sample_weight)

    predict_proba = model.predict_proba(X_test)
    proba = [x[1] for x in predict_proba]
    if class_weight == 'balanced':
        sample_weight = unbalanced_sample_weight(y_test)
    else:
        sample_weight = None
    loss = log_loss(y_test, proba, sample_weight=sample_weight)
    logger.debug('loss is %f', loss)

    return model, loss


def select_classifier(
        classifiers,
        X_train_val,
        y_train_val,
        n_splits=3,
        rebalance=False,
        random_state=None,
        ):
    logger.debug('selecting a classifier with X(%d, %d) and y(%d)',
                 *X_train_val.shape, len(y_train_val))

    method_losses = defaultdict(list)
    kf = model_selection.KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    fold_no = 0
    for train_idx, test_idx in kf.split(X_train_val):
        logger.debug("fold #%d", fold_no)
        fold_no += 1

        X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
        logger.debug("%d samples were selected for training", len(X_train))
        X_test, y_test = X_train_val[test_idx], y_train_val[test_idx]
        logger.debug("%d samples were selected for validation", len(X_test))

        for method in classifiers:
            _, loss = do_model(clone(method), X_train, y_train, X_test, y_test,
                               class_weight='balanced' if rebalance else None)
            method_losses[method].append(loss)

    model, _ = min(method_losses.items(), key=lambda x: np.mean(x[1]))
    logger.info("The best model according to minimal mean log-loss is %s", model)

    logger.info('re-training the selected model with all data...')
    model.fit(X_train_val, y_train_val)

    return model


def select_model(
        X_train_val,
        y_train_val,
        ):
    n_splits = 3
    max_depths = (10, 20, 30, 40)
    random_state = None

    methods = [NaiveBinaryClassifier(), BinaryLogisticRegression()]
    #for max_depth in range(10, 21):
    #    methods.append(Classifier(tree.DecisionTreeClassifier, max_depth=max_depth, min_samples_split=100))
    for max_depth in max_depths:
        methods.append(
            RandomForestBinaryClassifier(max_depth=max_depth)
        )

    methods.append(
        GradientBoostingBinaryClassifier()
    )

    model = select_classifier(
        methods,
        X_train_val, y_train_val,
        n_splits=n_splits,
        rebalance=True,
        random_state=random_state)

    return model

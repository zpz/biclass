from biclass.binary_classifier import (
    DecisionTreeBinaryClassifier,
    RandomForestBinaryClassifier
)

X = [[0, 0], [1, 1]]
y = [0, 1]
xx = [[2., 2.]]


def test_DecisionTreeBinaryClassifier():
    clf = DecisionTreeBinaryClassifier()
    clf = clf.fit(X, y)
    print('')
    print(clf)

    yhat = clf.predict(xx)
    print(yhat)

    p = clf.predict_proba(xx)
    print(p)


def test_RandomForestBinaryClassifier():
    clf = RandomForestBinaryClassifier(n_estimators=10)
    clf = clf.fit(X, y)
    print('')
    print(clf)

    yhat = clf.predict(xx)
    print(yhat)

    p = clf.predict_proba(xx)
    print(p)


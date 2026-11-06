from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=500,
        tol=1e-3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model
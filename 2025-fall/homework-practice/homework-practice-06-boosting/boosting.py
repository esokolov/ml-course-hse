from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from typing import Optional
from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []
        self.gammas = []

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose
        self.classes_ = np.array([-1, 1])  # в нашей задаче классы захардкожены

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: z * np.exp(np.log(y))  # Исправьте формулу на правильную.

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        raise Exception("partial_fit method not implemented")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        """
        train_predictions = np.zeros(X_train.shape[0])

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for _ in estimator_range:
            self.partial_fit(...)

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray):
        raise Exception("predict_proba method not implemented")

    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])


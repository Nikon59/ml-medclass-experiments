from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier


class RandomForestModel(BaseClassifier):
    """
    Случайный лес для мультиклассовой классификации.

    Наследуется от BaseClassifier и создаёт полиморфный отчёт обучения
    с блоком ensemble_analysis.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        criterion: str = 'gini',
        class_weight: Optional[str] = None,
        oob_score: bool = False,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            class_weight=class_weight,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=-1,
        )

        self._avg_tree_depth: Optional[float] = None
        self._max_tree_depth: Optional[int] = None
        self._avg_tree_leaves: Optional[float] = None
        self._oob_score_value: Optional[float] = None
        self._n_features: Optional[int] = None
        self.feature_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> 'RandomForestModel':
        """
        Обучает случайный лес. eval_set игнорируется (леса не поддерживают
        early stopping).
        """
        self.model.fit(X, y)

        self._n_features = X.shape[1]

        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        tree_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]

        self._avg_tree_depth = float(np.mean(tree_depths))
        self._max_tree_depth = int(max(tree_depths))
        self._avg_tree_leaves = float(np.mean(tree_leaves))

        if self.oob_score:
            self._oob_score_value = float(self.model.oob_score_)

        self.feature_importance = self.model.feature_importances_

        self._build_training_report()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def _build_training_report(self) -> None:
        self._training_report = {
            'type': 'random_forest',
            'feature_importance': (
                self.feature_importance.tolist()
                if self.feature_importance is not None else None
            ),
            'n_features_in': self._n_features,
            'ensemble_analysis': {
                'n_estimators': self.n_estimators,
                'max_depth_configured': self.max_depth,
                'avg_tree_depth': self._avg_tree_depth,
                'max_tree_depth': self._max_tree_depth,
                'avg_tree_leaves': self._avg_tree_leaves,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'criterion': self.criterion,
                'class_weight': self.class_weight,
                'oob_score': self._oob_score_value,
            },
        }

    def get_model_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'class_weight': self.class_weight,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
        }
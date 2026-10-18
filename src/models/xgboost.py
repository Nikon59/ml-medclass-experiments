from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import xgboost as xgb

from .base import BaseClassifier


def _subsample_history(history: Dict, step: int = 5) -> Dict:
    """Уменьшает размер истории обучения в N раз для экономии места в JSON."""
    result = {}
    for dataset_name, metrics in history.items():
        result[dataset_name] = {}
        for metric_name, values in metrics.items():
            result[dataset_name][metric_name] = values[::step]
    return result


class XGBoostModel(BaseClassifier):
    """
    XGBoost для мультиклассовой классификации.

    Требования:
    - y должен быть закодирован как 0, 1, ..., N-1 (через LabelEncoder)
    - XGBoost воспринимает int-признаки как непрерывные (важно!)
    """

    def __init__(
            self,
            n_estimators: int = 200,
            max_depth: int = 6,
            learning_rate: float = 0.1,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 1.0,
            early_stopping_rounds: int = 30,
            random_state: int = 42,
            num_class: int = 3
    ):
        super().__init__()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.num_class = num_class

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_class,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False
        )

        self._raw_evals: Optional[Dict] = None
        self._best_iteration: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'XGBoostModel':
        fit_params = {'verbose': False}

        if eval_set is not None and len(eval_set) > 0:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False

        self.model.fit(X, y, **fit_params)

        # XGBoost сохраняет evals_result даже без early_stopping
        if eval_set is not None and len(eval_set) > 0:
            self._raw_evals = self.model.evals_result()
            self._best_iteration = getattr(self.model, 'best_iteration', self.n_estimators - 1)
        else:
            self._raw_evals = None
            self._best_iteration = self.n_estimators - 1

        self.feature_importance = self.model.feature_importances_
        self._build_training_report()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def _build_training_report(self) -> None:
        early_stopped = False
        n_estimators_actual = self.n_estimators
        overfitting_gap = 0.0

        if self._raw_evals is not None:
            # XGBoost: {'validation_0': {'mlogloss': [...]}}
            valid_key = next(iter(self._raw_evals.keys()), None)
            if valid_key is not None and 'mlogloss' in self._raw_evals[valid_key]:
                full_length = len(self._raw_evals[valid_key]['mlogloss'])
                n_estimators_actual = full_length

                if self._best_iteration is not None:
                    early_stopped = self._best_iteration < self.n_estimators - 1

                    # Получаем train и validation метрики
                    train_key = 'validation_0' if 'validation_0' not in self._raw_evals else None
                    # Для XGBoost eval_set передается как [(X_test, y_test)]
                    # Ключи: 'validation_0' (первый eval_set), и т.д.
                    # Без train в eval_set, у нас только валидация
                    valid_final = self._raw_evals[valid_key]['mlogloss'][-1]
                    overfitting_gap = 0.0  # Нельзя вычислить без train в eval_set

        report: Dict[str, Any] = {
            'type': 'xgboost',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': int(self.feature_importance.shape[0]) if self.feature_importance is not None else 0,
            'booster_analysis': {
                'n_estimators_requested': self.n_estimators,
                'n_estimators_actual': n_estimators_actual,
                'early_stopped': early_stopped,
                'best_iteration': self._best_iteration,
                'overfitting_gap': overfitting_gap,
                'tree_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'num_class': self.num_class
            }
        }

        if self._raw_evals is not None:
            report['booster_analysis']['evals_result'] = _subsample_history(self._raw_evals, step=5)
            # Получаем полную длину истории
            for dataset_metrics in self._raw_evals.values():
                for metric_values in dataset_metrics.values():
                    if isinstance(metric_values, list):
                        report['booster_analysis']['evals_result_full_length'] = len(metric_values)
                        break
                break

        self._training_report = report

    def get_model_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'num_class': self.num_class
        }
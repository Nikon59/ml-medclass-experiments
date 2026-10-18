from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import lightgbm as lgb

from .base import BaseClassifier


def _subsample_history(history: Dict, step: int = 5) -> Dict:
    """Уменьшает размер истории обучения в N раз для экономии места в JSON."""
    result = {}
    for dataset_name, metrics in history.items():
        result[dataset_name] = {}
        for metric_name, values in metrics.items():
            result[dataset_name][metric_name] = values[::step]
    return result


class LightGBMModel(BaseClassifier):
    """
    LightGBM для мультиклассовой классификации.

    Требования:
    - y должен быть закодирован как 0, 1, ..., N-1
    - num_class = количество уникальных классов
    """

    def __init__(
            self,
            n_estimators: int = 200,
            max_depth: int = -1,
            learning_rate: float = 0.1,
            num_leaves: int = 31,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 0.0,
            min_child_samples: int = 20,
            early_stopping_rounds: int = 30,
            random_state: int = 42,
            num_class: int = 3
    ):
        super().__init__()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.num_class = num_class

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            objective='multiclass',
            metric='multi_logloss',
            num_class=num_class,
            random_state=random_state,
            verbosity=-1
        )

        self._raw_evals: Optional[Dict] = None
        self._best_iteration: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'LightGBMModel':
        fit_params = {}

        if eval_set is not None and len(eval_set) > 0:
            fit_params['eval_X'] = eval_set[0][0]
            fit_params['eval_y'] = eval_set[0][1]

            fit_params['callbacks'] = [
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=0)
            ]

        self.model.fit(X, y, **fit_params)

        # LightGBM сохраняет evals_result_ как атрибут
        if eval_set is not None and len(eval_set) > 0 and hasattr(self.model, 'evals_result_'):
            self._raw_evals = self.model.evals_result_
            self._best_iteration = getattr(self.model, 'best_iteration_', self.n_estimators)
        else:
            self._raw_evals = None
            self._best_iteration = self.n_estimators

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
            # LightGBM: {'valid_0': {'multi_logloss': [...]}}
            valid_key = next(iter(self._raw_evals.keys()), None)
            if valid_key is not None and 'multi_logloss' in self._raw_evals[valid_key]:
                full_length = len(self._raw_evals[valid_key]['multi_logloss'])
                n_estimators_actual = full_length

                if self._best_iteration is not None:
                    early_stopped = self._best_iteration < self.n_estimators

        report: Dict[str, Any] = {
            'type': 'lightgbm',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': int(self.feature_importance.shape[0]) if self.feature_importance is not None else 0,
            'booster_analysis': {
                'n_estimators_requested': self.n_estimators,
                'n_estimators_actual': n_estimators_actual,
                'early_stopped': early_stopped,
                'best_iteration': self._best_iteration,
                'overfitting_gap': overfitting_gap,
                'tree_depth': self.max_depth,
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'min_child_samples': self.min_child_samples,
                'num_class': self.num_class
            }
        }

        if self._raw_evals is not None:
            report['booster_analysis']['evals_result'] = _subsample_history(self._raw_evals, step=5)
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
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_samples': self.min_child_samples,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'num_class': self.num_class
        }
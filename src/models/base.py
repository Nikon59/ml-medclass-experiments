from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class BaseClassifier(ABC):
    """Базовый класс с поддержкой eval_set для early stopping."""

    def __init__(self):
        self.model: Optional[Any] = None
        self._training_report: Optional[Dict[str, Any]] = None

    @abstractmethod
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'BaseClassifier':
        """Обучает модель. eval_set используется для early stopping в бустингах."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    def get_training_history(self) -> Dict[str, Any]:
        """Возвращает полиморфный отчет об обучении."""
        if self._training_report is None:
            return {'type': 'unknown'}
        return self._training_report
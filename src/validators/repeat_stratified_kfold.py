from sklearn.model_selection import RepeatedStratifiedKFold
from typing import Iterator, Tuple, Optional, Any
import numpy as np

from src.validators.base import BaseValidator


class RepeatedStratifiedKFoldValidator(BaseValidator):
    """
    Повторяет стратифицированную k-fold несколько раз с разными random_state.
    Дает более стабильные оценки для маленьких выборок.
    """

    def __init__(
            self,
            n_splits: int = 5,
            n_repeats: int = 3,
            random_state: int = 42
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.validator = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Генерирует индексы для train/test фолдов."""
        # groups для RepeatedStratifiedKFold не используется, но
        # принимается для совместимости со sklearn API.
        return self.validator.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Возвращает общее количество разбиений."""
        return self.n_splits * self.n_repeats

    def get_split_info(self) -> dict:
        """Возвращает информацию о валидаторе."""
        return {
            'type': 'RepeatedStratifiedKFold',
            'n_splits': self.n_splits,
            'n_repeats': self.n_repeats,
            'random_state': self.random_state,
            'total_folds': self.get_n_splits()
        }
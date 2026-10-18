from typing import Generator, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .base import BaseValidator


class StratifiedKFoldValidator(BaseValidator):
    """Валидатор на основе StratifiedKFold."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.validator = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Генерирует индексы для train/test фолдов."""
        # groups для StratifiedKFold не используется, но принимается
        # для совместимости со sklearn API.
        for train_idx, test_idx in self.validator.split(X, y, groups):
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Возвращает количество фолдов."""
        return self.n_splits

    def get_split_info(self) -> Dict[str, Any]:
        """Возвращает информацию о стратегии."""
        return {
            'strategy': 'StratifiedKFold',
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state
        }
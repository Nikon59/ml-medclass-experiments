from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Dict, Any, Optional
import numpy as np


class BaseValidator(ABC):
    """Базовый класс для всех стратегий кросс-валидации."""

    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Генерирует индексы для train/test фолдов.

        Сигнатура совместима со sklearn-сплиттерами:
        sklearn вызывает cv.split(X, y, groups=...), поэтому groups
        (и любые дополнительные kwargs) должны приниматься, даже если
        конкретная стратегия их игнорирует.

        Yields:
            Tuple[train_indices, test_indices]
        """
        pass

    @abstractmethod
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Возвращает количество фолдов."""
        pass

    @abstractmethod
    def get_split_info(self) -> Dict[str, Any]:
        """Возвращает метаинформацию о стратегии валидации."""
        pass
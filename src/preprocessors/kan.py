from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class KANPreprocessor:
    """
    Препроцессор, который превращает сырой DataFrame с категориями и пропусками
    в числовой нормированный DataFrame для KAN.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        ordinal_features: Optional[List[str]] = None,
        ordinal_categories: Optional[Dict[str, List[str]]] = None,
        missing_value: str = "отсутствует",
    ):
        self.categorical_features_input = list(categorical_features or [])
        self.ordinal_features_input = list(ordinal_features or [])
        self.ordinal_categories_input = ordinal_categories or {}
        self.missing_value = missing_value

        self.is_fitted = False

        self.feature_names_in_: List[str] = []
        self.categorical_features_: List[str] = []
        self.ordinal_features_: List[str] = []
        self.numeric_features_: List[str] = []

        self.cat_categories_: Dict[str, List[str]] = {}
        self.ordinal_maps_: Dict[str, Dict[str, int]] = {}

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None

        self.output_columns_: List[str] = []

    def _normalize_input_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.feature_names_in_:
            if col not in X.columns:
                X[col] = np.nan

        return X[self.feature_names_in_]

    def _clean_str_series(self, s: pd.Series) -> pd.Series:
        s = s.astype(str)
        s = s.replace(
            {
                "nan": self.missing_value,
                "None": self.missing_value,
                "": self.missing_value,
            }
        )
        return s

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "feature_names_in_", None):
            raise RuntimeError("KANPreprocessor: feature_names_in_ не задан.")

        X = self._normalize_input_columns(X)
        parts = []

        # 1. Категориальные признаки -> one-hot
        for col in self.categorical_features_:
            s = self._clean_str_series(X[col])

            categories = self.cat_categories_.get(col, [self.missing_value])
            known_categories = set(categories)

            s = s.where(s.isin(known_categories), self.missing_value)

            for cat in categories:
                parts.append(
                    pd.Series(
                        (s == cat).astype(float),
                        name=f"{col}={cat}",
                    )
                )

        # 2. Ordinal признаки -> порядковые коды
        for col in self.ordinal_features_:
            s = self._clean_str_series(X[col])

            mapping = self.ordinal_maps_.get(col, {self.missing_value: 0})
            missing_code = mapping.get(self.missing_value, len(mapping) - 1)

            codes = s.map(lambda v: mapping.get(v, missing_code))

            parts.append(
                pd.Series(
                    codes.astype(float),
                    name=col,
                )
            )

        # 3. Числовые признаки
        for col in self.numeric_features_:
            vals = pd.to_numeric(X[col], errors="coerce")

            parts.append(
                pd.Series(
                    vals.astype(float),
                    name=col,
                )
            )

        if not parts:
            raise ValueError(
                "После кодирования не осталось признаков. "
                f"X.shape={X.shape}, "
                f"X.columns[:10]={list(X.columns)[:10]}, "
                f"categorical_features={self.categorical_features_}, "
                f"ordinal_features={self.ordinal_features_}, "
                f"numeric_features={self.numeric_features_}"
            )

        out = pd.concat(parts, axis=1)
        out.columns = [str(c) for c in out.columns]
        out = out.replace([np.inf, -np.inf], np.nan)

        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        if X.shape[1] == 0:
            raise ValueError("В fit_transform передан пустой DataFrame без признаков.")

        # Сначала фиксируем входные признаки
        self.feature_names_in_ = list(X.columns)

        # Затем распределяем типы признаков
        self.categorical_features_ = [
            c for c in self.categorical_features_input if c in X.columns
        ]

        self.ordinal_features_ = [
            c
            for c in self.ordinal_features_input
            if c in X.columns and c not in self.categorical_features_
        ]

        self.numeric_features_ = [
            c
            for c in X.columns
            if c not in self.categorical_features_
               and c not in self.ordinal_features_
        ]

        if (
                not self.categorical_features_
                and not self.ordinal_features_
                and not self.numeric_features_
        ):
            raise ValueError(
                "KANPreprocessor не смог определить ни одного признака. "
                f"X.shape={X.shape}, X.columns={list(X.columns)}"
            )

        # Категориальные категории
        self.cat_categories_ = {}

        for col in self.categorical_features_:
            values = self._clean_str_series(X[col]).unique().tolist()
            categories = sorted(set(values + [self.missing_value]))
            self.cat_categories_[col] = categories

        # Ordinal mapping
        self.ordinal_maps_ = {}

        for col in self.ordinal_features_:
            base_categories = list(self.ordinal_categories_input.get(col, []))

            if len(base_categories) == 0:
                base_categories = self._clean_str_series(X[col]).unique().tolist()

            if self.missing_value not in base_categories:
                base_categories.append(self.missing_value)

            train_values = self._clean_str_series(X[col]).unique().tolist()

            extra_values = [
                v
                for v in sorted(set(train_values))
                if v not in base_categories and v != self.missing_value
            ]

            if extra_values:
                if self.missing_value in base_categories:
                    base_categories.remove(self.missing_value)

                base_categories = base_categories + extra_values + [self.missing_value]

            self.ordinal_maps_[col] = {
                category: i for i, category in enumerate(base_categories)
            }

        # Только теперь кодируем
        X_encoded = self._encode(X)

        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X_encoded)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.output_columns_ = list(X_encoded.columns)
        self.is_fitted = True

        return pd.DataFrame(X_scaled, columns=self.output_columns_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("KANPreprocessor сначала должен быть обучен.")

        X_encoded = self._encode(X)

        missing_cols = [
            c for c in self.output_columns_ if c not in X_encoded.columns
        ]

        for c in missing_cols:
            X_encoded[c] = 0.0

        X_encoded = X_encoded[self.output_columns_]
        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

        X_imputed = self.imputer.transform(X_encoded)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.output_columns_)
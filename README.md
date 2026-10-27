# ml-medclass-experiments

Классификация структурированных клинических медицинских данных (дифференциальная диагностика острого панкреатита: отёчный панкреатит / панкреонекроз). Репозиторий содержит исследовательские Jupyter Notebooks, модульную библиотеку компонентов (`src/`), интерфейс командной строки (`cli.py`), а также настроенный CI/CD-пайплайн для сборки автономных исполняемых пакетов под Linux и Windows.

---

## Структура репозитория

```text
.
├── .github/
│   └── workflows/
│       └── build-release.yml      # CI/CD: параллельная сборка релизов Linux и Windows
├── notebooks/                     # Исследовательские ноутбуки (EDA, гипотезы, отбор признаков)
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_grid_search.ipynb
│   ├── 04_learning.ipynb
│   ├── 05_correlated_traits.ipynb
│   ├── 06_top_feature.ipynb
│   ├── 07_comparison.ipynb
│   └── 08_export_import.ipynb
├── configs/
│   └── tuning.yaml                # Сетки гиперпараметров для оптимизации
├── data/
│   ├── raw/                       # Исходные медицинские протоколы и таблицы признаков
│   └── processed/                 # Сформированные выборки (dataset.parquet, schema.json)
├── results/                       # Артефакты: leaderboard.csv, best_params_*.json, .onnx, .joblib
├── src/                           # Модули пайплайна
│   ├── dataset/                   # Загрузка и валидация данных (loader.py, dataset_builder.py)
│   ├── models/                    # Duck-typing реестр алгоритмов (registry.py)
│   ├── metrics/                   # Расширяемый реестр метрик качества (registry.py)
│   ├── tuning/                    # Модуль подбора гиперпараметров (tuner.py)
│   ├── export/                    # Сериализация и валидация ONNX/Joblib (exporter.py)
│   ├── preprocessors/             # Препроцессинг и нормализация признаков
│   └── validators/                # Стратегии стратифицированной кросс-валидации
├── tests/                         # Модульные и интеграционные тесты на pytest
├── cli.py                         # Консольный интерфейс приложения
├── requirements.txt               # Зависимости Python
└── pytest.ini                     # Конфигурация запуска тестов
```

---

## Рабочий процесс

1. **Исследования в Jupyter Notebooks (`notebooks/`):**
   Разведочный анализ данных (EDA), статистическая проверка гипотез и отбор признаков ведутся в интерактивной среде ноутбуков.
2. **Фиксация данных (`data/processed/`):**
   Очищенные выборки сохраняются в формате Parquet (`dataset.parquet`) или CSV с обязательной генерацией схемы валидации типов (`schema.json`).
3. **Автоматизация через CLI (`cli.py`):**
   Консольные команды берут подготовленный датасет, запускают кросс-валидацию, подбор гиперпараметров, обучение моделей и экспорт графов в формат ONNX с проверкой численной идентичности выходов в `onnxruntime`.

---

## Установка и запуск

### Вариант А. Автономные бинарные пакеты (без установки Python)

Скомпилированные пакеты содержат все C++ и Python библиотеки (CatBoost, XGBoost, LightGBM, ONNX Runtime). Доступны на странице **[GitHub Releases](https://github.com/Nikon59/ml-medclass-experiments/releases)**.

* **Linux (x86_64):**
  ```bash
  tar -xzf medclass-cli-linux-x64.tar.gz
  ./medclass-cli/medclass-cli --help
  ```

* **Windows (x64):**
  ```powershell
  Expand-Archive medclass-cli-windows-x64.zip -DestinationPath .
  .\medclass-cli\medclass-cli.exe --help
  ```

> **Важно:** Запуск бинарника должен производиться из корневой директории проекта, где расположены каталоги `data/` и `configs/`.

---

### Вариант Б. Запуск из исходного кода (Python 3.10+)

```bash
# 1. Клонирование репозитория
git clone [https://github.com/Nikon59/ml-medclass-experiments.git](https://github.com/Nikon59/ml-medclass-experiments.git)
cd ml-medclass-experiments

# 2. Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate       # На Windows: .venv\Scripts\Activate.ps1

# 3. Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# 4. Проверка готовности системы
python cli.py list
```

---

## Использование CLI

> При использовании скомпилированного бинарника замените `python cli.py` на путь к утилите: `./medclass-cli/medclass-cli` (Linux) или `.\medclass-cli\medclass-cli.exe` (Windows).

Все команды используют значения по умолчанию: датасет `data/processed/dataset.parquet` (или `dataset.csv`), алгоритм `catboost` и базовый пул диагностических метрик.

### 1. Просмотр реестров (`list`)
Выводит список доступных моделей с их параметрами по умолчанию и перечень зарегистрированных метрик:
```bash
python cli.py list
```

### 2. Сборка датасета из сырых файлов (`prep-data`)
Преобразует сырые медицинские таблицы, очищает пропуски и строит схему признаков:
```bash
# Сборка по путям по умолчанию
python cli.py prep-data

# С явным указанием источников
python cli.py prep-data --input data/raw/algorithm.txt --output data/processed/dataset.parquet
```

### 3. Обучение и экспорт (`train`)
Обучает классификатор, считает метрики на тестовой выборке, сохраняет модель в Joblib и ONNX, после чего валидирует предсказания через ONNX Runtime (`np.testing.assert_allclose(py_probs, onnx_probs, atol=1e-5)`):
```bash
# Базовый запуск:
python cli.py train

# Обучение модели с параметрами из JSON-файла тюнинга:
python cli.py train --model logreg --params-file results/best_params_logreg.json

# Обучение с кастомными путями сохранения артефактов:
python cli.py train --model catboost --export-onnx results/catboost.onnx --export-joblib results/catboost.joblib
```

### 4. Сравнительный бенчмарк моделей (`benchmark`)
Запускает кросс-валидацию по всем алгоритмам реестра и формирует общую сводную таблицу качества:
```bash
# Бенчмарк со стандартными настройками:
python cli.py benchmark

# Детальный запуск по выбранным моделям и метрикам:
python cli.py benchmark \
  --models catboost xgboost lightgbm logreg random_forest \
  --metrics roc_auc f1_macro accuracy recall precision \
  --cv 5 \
  --output results/leaderboard.csv
```

### 5. Оптимизация гиперпараметров (`tune`)
Выполняет `GridSearchCV` или `RandomizedSearchCV` по сеткам из `configs/tuning.yaml`:
```bash
# Grid search для CatBoost:
python cli.py tune --model catboost

# Random search с ограничением количества итераций:
python cli.py tune --model logreg --search-type random --n-iter 20 --cv 5 --output results/
```
Лучшие параметры сохраняются в `results/best_params_<model>.json`.

### 6. Верификация ONNX-графа (`verify-onnx`)
Проверяет целостность графа, соответствие размерностей тензоров, замеряет скорость инференса и сверяет результаты с Joblib-сериализацией:
```bash
python cli.py verify-onnx --onnx results/catboost.onnx
```

---

## Руководство разработчика

### Регистрация новой модели
Система использует утиную типизацию (Scikit-Learn API: методы `fit`, `predict`, `predict_proba`). Модель подключается через декоратор `@register_model`:

```python
from sklearn.neighbors import KNeighborsClassifier
from src.models.registry import register_model

@register_model(
    name="knn",
    default_params={"n_neighbors": 5, "weights": "distance"}
)
class RegisteredKNN(KNeighborsClassifier):
    pass
```
После этого модель сразу становится доступна во всех командах CLI (`train`, `benchmark`, `tune`).

### Регистрация пользовательской метрики
Метрики добавляются через реестр `@register_metric`:

```python
import numpy as np
from src.metrics.registry import register_metric

@register_metric("error_rate")
def compute_error_rate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> float:
    return float(np.mean(y_true != y_pred))
```

---

## Тестирование и автоматическая сборка

Запуск набора тестов:
```bash
pytest -v tests/
```

Тестовый контур контролирует:
- Корректность duck-typing оберток всех классификаторов.
- Расчет метрик в бинарном и многоклассовом режимах.
- Экспорт в Joblib и ONNX с пороговой валидацией точности в ONNX Runtime.
- Выполнение всех команд CLI (`list`, `prep-data`, `train`, `benchmark`, `tune`, `verify-onnx`).

### Автоматический релиз бинарников (CI/CD)
Проект настроен на сборку кроссплатформенных артефактов через GitHub Actions:
* При отправке тега версии (`git tag v1.1.0 && git push origin v1.1.0`) запускается параллельная сборка под **Ubuntu 22.04** и **Windows Latest**.
* Скомпилированные архивы автоматически прикрепляются к созданному релизу на GitHub в раздел Assets.

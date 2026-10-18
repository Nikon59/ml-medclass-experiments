# ml-medclass-experiments

Исследовательский ML-проект: многоклассовая классификация медицинских случаев (диагноз острого панкреатита: `отёчный панкреатит` / `панкреонекроз`) по структурированным клиническим признакам. Включает EDA, статистический анализ признаков, сравнение моделей, кросс-валидацию и экспорт лучшей модели в ONNX.

## Структура проекта

```
.
├── 01_eda.ipynb                 # Разведочный анализ данных
├── 02_feature_analysis.ipynb    # Глубокий анализ признаков и статистические гипотезы
├── 03_grid_search.ipynb         # Подбор гиперпараметров
├── 04_learning.ipynb            # Обучение моделей
├── 05_correlated_traits.ipynb   # Анализ коррелирующих признаков
├── 06_top_feature.ipynb         # Анализ значимых признаков
├── 07_comparison.ipynb          # Сравнение финальных моделей
├── 08_export.ipynb              # Экспорт модели в ONNX
├── create_dataset.py            # Сборка датасета из сырых файлов
├── src/                         # Библиотека проекта
│   ├── dataset/                 # Сборка датасета и схема признаков
│   ├── preprocessors/           # Препроцессоры
│   ├── models/                  # Обёртки моделей с единым интерфейсом
│   ├── validators/              # Стратегии кросс-валидации
│   ├── pipeline.py              # Фасад «препроцессор + модель», экспорт в ONNX
│   ├── evaluator.py             # Оценка качества (CV + holdout, ROC/PR)
│   ├── tuner.py                 # GridSearchTuner
│   └── runner.py                # ExperimentRunner — оркестрация экспериментов
├── data/
│   ├── raw/                     # Сырые данные (не изменяются)
│   └── processed/               # dataset.csv + dataset.schema.json
├── results/                     # Артефакты экспериментов (создаётся при запуске)
└── requirements.txt
```

## Данные

Сырые данные (`data/raw/`, разделитель `;`, кодировка UTF-8):

| Файл | Содержание |
|---|---|
| `features.csv` | Описание признаков: ID, имя, тип (кат./ран.), диапазоны значений |
| `value_features.csv` | Расшифровка числовых кодов значений признаков |
| `algorithm.txt` | Матрица наблюдений: строки — пациенты, колонки — признаки; первое значение каждой строки — код диагноза (целевая переменная), `-1` — пропуск |

Признаки — категориальные и ординальные (пол, возраст, жалобы, анамнез, лабораторные показатели и т.д.). Целевая переменная — `diagnosis`.

### Сборка датасета

```bash
python create_dataset.py
```

Скрипт преобразует сырые файлы в `data/processed/dataset.csv` (значения заменены на человекочитаемые метки) и `data/processed/dataset.schema.json` (типы признаков и категории, включая явный маркер пропусков `MISSING`). Исключаемые признаки задаются в списке `excluded_features`.

## Библиотека (`src/`)

- **`dataset/`** — `DatasetBuilder` собирает датафрейм и схему из сырых файлов; `DatasetStore` / `SchemaConfig` — загрузка и конфигурация.
- **`preprocessors/`** — семейство препроцессоров с единым sklearn-совместимым интерфейсом: one-hot кодирование, one-hot + скейлинг, фильтрующий отбор признаков, бутстрап-обёртка (`booster`), а также вариант с KAN-расширением.
- **`models/`** — обёртки над моделями с единым API `fit / predict / predict_proba / get_training_history`:
  DecisionTree, RandomForest, LogisticRegression, Ridge, SGD, ComplementNB, SVM, LightGBM, XGBoost.
- **`validators/`** — стратифицированный K-Fold и повторяющийся стратифицированный K-Fold.
- **`pipeline.py`** — класс `Pipeline(preprocessor, model)`: sklearn-совместимый фасад, фиксирует `feature_names_in_/out_`, умеет `predict/predict_proba` и `export_onnx()` (через `skl2onnx`, принимает строковые признаки; рядом пишет `meta.json` с описанием входа).
- **`evaluator.py`** — `ModelEvaluator`: CV- и holdout-метрики качества классификации.
- **`tuner.py`** — `GridSearchTuner`: перебор сетки гиперпараметров с валидацией.
- **`runner.py`** — `ExperimentRunner`: оркестратор экспериментов. Каждый опыт сохраняется в `results/<name>/`:
  - `result.json` — CV + holdout метрики, конфиг, время обучения;
  - `best_detailed.json` — детальные результаты grid search;
  - `pipeline.joblib` — обученный пайплайн (лучший для grid search).

## Ноутбуки

1. **01_eda** — распределения, пропуски, баланс классов.
2. **02_feature_analysis** — статистические гипотезы и значимость признаков.
3. **03_grid_search** — пример подбора гиперпараметров кандидатов.
4. **04_learning** — пример обучения модели.
5. **05_correlated_traits** — сравнение влияния коррелирующих признаков.
6. **06_top_feature** — сравнение влияния топ-признаков.
7. **07_comparison** — сравнение моделей
8. **08_export_import** — экспорт выбранной модели в ONNX и проверка.

Запуск Jupyter:

```bash
jupyter lab
```

## Быстрый старт

```bash
# 1. Окружение (Python >= 3.9)
python -m venv .venv
source .venv/bin/activate

# 2. Зависимости
pip install -r requirements.txt

# 3. Сборка датасета
python create_dataset.py

# 4. Ноутбуки / эксперименты
jupyter lab
```
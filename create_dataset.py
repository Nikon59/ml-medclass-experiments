from pathlib import Path

from src.dataset import DatasetBuilder


def main():
    # ------------------------------------------------------------------
    # 1. Пути к исходным файлам
    # ------------------------------------------------------------------
    features_file = "data/raw/features.csv"               # описание признаков
    value_features_file = "data/raw/value_features.csv"   # расшифровка значений
    data_file = "data/raw/algorithm.txt"                   # сырые данные

    # ------------------------------------------------------------------
    # 2. Параметры сборки
    # ------------------------------------------------------------------
    target_name = "diagnosis"          # имя целевой колонки
    output_csv = "data/processed/dataset.csv"      # куда сохранить датафрейм
    output_schema = "data/processed/dataset.schema.json"  # куда сохранить схему

    # Признаки, которые нужно исключить
    excluded_features = [
        "Ds",
        "Corse",
        "effect of treat"
    ]

    use_value_labels = True

    # ------------------------------------------------------------------
    # 3. Сборка и сохранение
    # ------------------------------------------------------------------
    builder = DatasetBuilder(
        features_file=features_file,
        value_features_file=value_features_file,
        data_file=data_file,
        target_name=target_name,
    )

    builder.save_dataset(
        filepath=output_csv,
        schema_path=output_schema,
        excluded_features=excluded_features,
        use_value_labels=use_value_labels,
        sep=';',
        encoding='utf-8',
    )

    print(f"Датасет сохранён: {output_csv}")
    print(f"Схема сохранена:  {output_schema}")


if __name__ == "__main__":
    main()
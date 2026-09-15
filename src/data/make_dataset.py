# -*- coding: utf-8 -*-
"""Script principal para construir datasets limpios y preparados."""

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from services.data_cleaning.config import CleaningConfig
from services.data_cleaning.pipeline import CleaningPipeline
from services.feature_engineering.config import FeatureConfig
from services.feature_engineering.pipeline import FeaturePipeline


def make_dataset():
    # Cargar variables de entorno
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logger.info("Iniciando construcción y preparación de datasets")

    # Ejecutar pipeline de limpieza
    cleaning_config = CleaningConfig()
    cleaning_pipeline = CleaningPipeline(cleaning_config)
    cleaning_result = cleaning_pipeline.run()

    # Ejecutar pipeline de ingeniería de características
    feature_config = FeatureConfig()
    feature_pipeline = FeaturePipeline(feature_config)
    feature_result = feature_pipeline.run(
        cleaning_result["df_validas"],
        stats_cleaning=cleaning_result["stats"],
    )

    logger.info("Proceso de preparación finalizado exitosamente.")
    return feature_result


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    make_dataset()

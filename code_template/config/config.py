from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"

#not in use
import tomli
with open("config.toml", mode="rb") as fp:
    toml_config = tomli.load(fp)



# Since we're using the MLflowCallback here with Optuna, we can either allow all our experiments to be stored under the default mlruns directory that MLflow will create or we can configure that location:
import mlflow
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

#Logging
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import sys
import logging
import logging.config
from rich.logging import RichHandler

# Use config file to initialize logger
logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # set rich handler


#https://madewithml.com/courses/mlops/logging/
#moved to logging.config file
# logging_config = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "minimal": {"format": "%(message)s"},
#         "detailed": {
#             "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
#         },
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "stream": sys.stdout,
#             "formatter": "minimal",
#             "level": logging.DEBUG,
#         },
#         "info": {
#             "class": "logging.handlers.RotatingFileHandler",
#             "filename": Path(LOGS_DIR, "info.log"),
#             "maxBytes": 10485760,  # 1 MB
#             "backupCount": 10,
#             "formatter": "detailed",
#             "level": logging.INFO,
#         },
#         "error": {
#             "class": "logging.handlers.RotatingFileHandler",
#             "filename": Path(LOGS_DIR, "error.log"),
#             "maxBytes": 10485760,  # 1 MB
#             "backupCount": 10,
#             "formatter": "detailed",
#             "level": logging.ERROR,
#         },
#     },
#     "root": {
#         "handlers": ["console", "info", "error"],
#         "level": logging.INFO,
#         "propagate": True,
#     },
# }
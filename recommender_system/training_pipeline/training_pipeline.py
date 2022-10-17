import warnings
from dataclasses import dataclass

from loguru import logger

from data_hadler import DataHandler
from model_trainer import ModelTrainer

warnings.simplefilter(action="ignore", category=FutureWarning)


data_handler = DataHandler()
logger.info(data_handler.config)
data_handler.run()
data_handler.save_data()

model_trainer = ModelTrainer(data_handler.whole_dataset)
model_trainer.train()
model_trainer.save()

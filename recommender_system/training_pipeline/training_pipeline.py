import warnings
from dataclasses import dataclass

from loguru import logger

from data_hadler import DataHandler
from model_trainer import ModelTrainer


warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    data_handler = DataHandler()
    logger.info(f"Using the following configuration: {data_handler.config}")
    data_handler.run()
    data_handler.save_data()

    model_trainer = ModelTrainer(data_handler.whole_dataset)
    model_trainer.train()
    model_trainer.save()


if __name__ == "__main__":
    main()

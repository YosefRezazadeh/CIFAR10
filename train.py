from src.trainer import Trainer
from src.utilities import config_parser
import argparse

if __name__ == "__main__" :
 
   parser = argparse.ArgumentParser()
   parser.add_argument('--config_path', type=str)

   args = parser.parse_args()
   config_file_path = args.config_path

   config = config_parser(config_file_path)

   trainer = Trainer(config['model'], config['dataloaders'], config['optimzer'], config['scheduler'], config['loss_function'])

   trainer.train(config['epochs'], config['save_config'], config['config'], config['device'])
   
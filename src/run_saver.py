import subprocess
import torch
from .train_logger import TrainLogger
import matplotlib.pyplot as plt
import os
import json
from .torch_summary import summary_string

class RunSaver:
   """
   Save log and information of a train
   
   Attributes:
      new_code (int): ID of new train directory
      new_dir_name (str): Name of new train directory
      logger (TrainLogger): TrainLogger instance for new train log file
   """

   def __init__(self):
      """
      Initialize saver consist of creating run directory and counting number existing train directories
      """
      
      if not os.path.exists('./runs/') :
         os.system('mkdir ./runs/')
      
      # count number existing train directories
      dirs_count = int(subprocess.check_output('ls -d ./runs/train-* | wc -l', shell=True).decode().strip())
      
      self.new_code = dirs_count
      
   def prepare_new_train_save(self):
      """
      Prepare a new directory for new train
      
      """
      self.new_code += 1
      
      self.new_dir_name = f'train-{self.new_code}'
      os.system(f'mkdir ./runs/{self.new_dir_name}')
      os.system(f'mkdir ./runs/{self.new_dir_name}/weights')
      os.system(f'mkdir ./runs/{self.new_dir_name}/checkpoints')
      os.system(f'touch ./runs/{self.new_dir_name}/log.txt')
      
      self.logger = TrainLogger(f'./runs/{self.new_dir_name}/log.txt')
      
   def save_batch_images(self, images, cols, rows, batch_id):
      """
      Save selected part of images of a batch in a grid as png file
      
      Args:
         images (torch.Tensor): A batch of images in format of (B,C,H,W)
         cols (int): Columns of grid
         rows (int): Rows of grid
         batch_id (int): ID of batch for image file name
         
      """
      figure = plt.figure(figsize=(12, 12))

      for i in range(1, cols * rows + 1):
         img = (images[i-1] + 1) * 127.5  # do inverse of preprocess 
         figure.add_subplot(rows, cols, i)
         plt.axis("off")
         plt.imshow(img.permute(1,2,0).type(torch.uint8))
      
      plt.savefig(f'./runs/{self.new_dir_name}/batch-{batch_id}.png')
   
   def save_model(self, model, name):
      """
      Save given model in current train weights directory
      
      Args:
         model (torch.nn.Module): Model to be saved
         name (str): File name of model
         
      """
      torch.save(model, f'./runs/{self.new_dir_name}/weights/{name}.pth')
      
   def save_config(self, config_dict):
      """
      Save given config dictionary as json file
      
      Args:
         save_config (dict): Config dictionary
         
      """
      with open(f'./runs/{self.new_dir_name}/config.json', "w") as config_file:
         json.dump(config_dict, config_file, indent=3)
         
   def save_model_summary(self, model):
      """
      Print torch model in a text file
      
      Args:
         model (torch.nn.Module): Model to be summerized
         
      """
      with open(f'./runs/{self.new_dir_name}/model_summary.txt', "w") as model_summary_file:
         model_summary_file.write(summary_string(model, (3,32,32))[0])
         
   def get_saved_models(self):
      """
      Returns list of saved model in training process
      
      Returns:
         saved_models (list): List of torch.nn.Module that loaded via torch.load
      """
      saved_models = []
      
      for model_file in os.listdir(f'./runs/{self.new_dir_name}/weights/'):
         saved_models.append((torch.load(f'./runs/{self.new_dir_name}/weights/{model_file}'), model_file.split('.')[0]))
      
      return saved_models
      
   def save_checkpoint(self, state_dict):
       """
       Save a checkpoint with given state dictionary
       
       Args:
          state_dict (dict): A dictionary which consist of model, optimizer, ... states
       """
       torch.save(state_dict, f'./runs/{self.new_dir_name}/checkpoints/checkpoint.pth')
      
      
      
import torch
import torch.nn as nn
import numpy as np
import json
import torchvision.transforms as transforms
from models import *
from .cifar_dataloader import CifarDataloader
from torcheval.metrics import MulticlassAccuracy
from tqdm import tqdm

def get_device():
   """
   Check our device (CPU/GPU)
   
   Returns:
      device (str): detected device
   """
   device = "cuda" if torch.cuda.is_available() else "cpu"

   return device


def sum_squard_weights(model, desired_layers=[nn.Conv2d, nn.Linear]):
   """
   Calculate sum squared of weights of given model

   Args:
      model (torch.nn.Module): Pytorch model
      desired_layers (list): List of pytorch layers
      
   Returns:
      One item Tensor as sum of squared weights of desired layers
   """
   sum = 0

   for layer in model.modules():
      flag = False

      for m in desired_layers :
         if isinstance(layer, m) :
            flag = True
            break

      if flag :
         sum += (layer.weight ** 2).sum()

   return sum.item()


def sum_squared_weights_checker(model, test_count):
   """
   Calculate mean sum squared of weights of given model and test count repeat

   Args:
      model (torch.nn.Module): Pytorch model
      test_count (int): numbers of tests
      
   Returns:
      One item numpy array as mean 
   """
   squared_weights = []

   for i in range(test_count) :
      squared_weights.append(sum_squard_weights(model))

   return np.mean(np.array(squared_weights))
   
   
def crossentropy_loss(label, pred, epsilon=1e-3):
    """
    Calculater Cross Entropy loss value
    
    Args:
       label (torch.Tensor): Labels tensor
       pred (torch.Tensor): Predictions tensor
       epsilon (float): Threshold to prevent log 0
       
    Returns:
       One item tensor as loss value
    """
    pred = torch.where(pred > epsilon, pred, epsilon)
    loss_value = torch.sum(-1 * label * torch.log(pred), axis=1)
    return loss_value
   
   
def config_parser(config_file_path):
   """
   Parse given config file (json file)
   
   Args:
      config_file_path (str): Json config file path
      
   Returns:
      built_config (dict): Dictionary of training components
   """
   json_file = open(config_file_path)
   config = json.load(json_file)
   
   # reading model config
   model_config = config["model_config"]
   
   # model activation function & bias and creat model
   activation_function_type = model_config["activation_function"]
   activation_function = torch.nn.ReLU()
   slope = 0
   if activation_function_type == 'leaky_relu' :
      slope = model_config["activation_function_negative_slope"]
      activation_function = torch.nn.LeakyReLU(slope)
      
   conv_use_bias = model_config["use_bias"]
   
   # model = Resnet(n, activation_function, conv_use_bias)
   if model_config["model"] == "resnet" :
      model = Resnet(architect_list=model_config["architect"], filter_list=model_config["filters"], activation_function=activation_function ,use_bias=conv_use_bias)
   elif model_config["model"] == "mobilenet" :
      model = Mobilenet(architect_list=model_config["architect"], filter_list=model_config["filters"], use_bias=conv_use_bias)
   elif model_config["model"] == "mobilenet-v2" :
      model = MobilenetV2(architect_list=model_config["architect"], filter_list=model_config["filters"], use_bias=conv_use_bias)
   
   # reading train config
   train_config = config["train_config"]
   
   # device config (cpu, cuda, auto)
   device_dict = {
      'train' : train_config["device"] if train_config["device"] in ['cpu','cuda'] else get_device(),
      'eval' : config["evaluation_config"]["device"] if config["evaluation_config"]["device"] in ['cpu','cuda'] else get_device()
   }
   model = model.to(device_dict["train"])
   
   # resume
   resume_config = train_config["resume"]
   use_resume = resume_config["use"]
   
   if use_resume :
      checkpoint = torch.load(resume_config["checkpoint_path"], map_location=device_dict["train"])
   
   # initialize model weights  
   if not use_resume :
      # using intializer
      initilizer_config = model_config["initializer_config"]
      initilizer_name = initilizer_config["initilizer"]
      initializer = None
       
      if initilizer_name == "kaiming_uniform" :
         initializer = torch.nn.init.kaiming_uniform_
         a, mode, non_linearity = slope, initilizer_config["mode"], activation_function_type
         model.set_initializer(initializer, a=a, mode=mode, nonlinearity=non_linearity)
      elif initilizer_name == "kaiming_normal" :
         initializer = torch.nn.init.kaiming_normal_
         a, mode, non_linearity = slope, initilizer_config["mode"], activation_function_type
         model.set_initializer(initializer, a=a, mode=mode, nonlinearity=non_linearity)
      elif initilizer_name == "xavier_uniform" :
         initializer = torch.nn.init.xavier_uniform_
         gain= initilizer_config["gain"]
         model.set_initializer(initializer, gain=gain)
      elif initilizer_name == "xavier_normal" :
         initializer = torch.nn.init.xavier_normal_
         gain= initilizer_config["gain"]
         model.set_initializer(initializer, gain=gain) 
   else:
      # load model weights from checkpoint
      model.load_state_dict(checkpoint['model_state_dict'])
    
   # reading data config
   data_config = train_config["data_config"]
   
   batch_size = data_config["batch_size"]
   dataset_path = data_config["dataset_path"]
   data_split = data_config["data_split"]
   
   # data augmentation
   data_augmentations = data_config["data_augmentation"]
   used_augmentations = []
   
   for aug_name in data_augmentations.keys():
      if data_augmentations[aug_name]["use"] :
         new_dict_aug = data_augmentations.get(aug_name)
         new_dict_aug['name'] = aug_name
         used_augmentations.append(new_dict_aug)
 
   augmentation_transforms = []
   for use_aug in used_augmentations :
      augmentation_config = use_aug
      use_aug_name = use_aug["name"] 
      if use_aug_name == "pad" :
         padding = augmentation_config["padding"]
         augmentation_transforms.append(transforms.Pad(padding))
      elif use_aug_name == "hflip" :
         p = augmentation_config["probability"]
         augmentation_transforms.append(transforms.RandomHorizontalFlip(p))
      elif use_aug_name == "crop" :
         size = augmentation_config["size"]
         augmentation_transforms.append(transforms.RandomCrop(size))
      elif use_aug_name == "color" :
         brightness = augmentation_config["brightness"]
         contrast = augmentation_config["contrast"]
         saturation = augmentation_config["saturation"]
         hue = augmentation_config["hue"]
         augmentation_transforms.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
         
   # evaluation config
   evaluation_batch_size = config["evaluation_config"]["batch_size"] 
         
   # prepare dataloaders
   cifar_dataloader = CifarDataloader(dataset_path, batch_size, evaluation_batch_size, augmentation_transforms)
         
   # optimizer
   optimizer_config = train_config["optimizer_config"]
   
   optimizer = None
   
   if optimizer_config["optimzer"] == "SGD" :
      momentum = optimizer_config["momentum"]
      weight_decay = optimizer_config["weight_decay"]
      init_lr = optimizer_config["initial_lr"]
      optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=2*weight_decay)
   elif optimizer_config["optimzer"] == "ADAM" :
      momentum = optimizer_config["momentum"]
      weight_decay = optimizer_config["weight_decay"]
      init_lr = optimizer_config["initial_lr"]
      beta_1, beta_2 = optimizer_config["beta_1"], optimizer_config["beta_2"]
      optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=2*weight_decay, betas=(beta_1, beta_2))
   
   if use_resume :
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

   # scheduler
   epochs = train_config["epochs"]
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, 3*epochs//4], gamma=0.1)
   if use_resume :
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      
   # config summary
   config_summary = {}
   config_summary['model_config'] = model_config
   config_summary['train_config'] = train_config
   config_summary['evaluation_config'] = config["evaluation_config"]
      
   # config dictionary
   built_config = {
      'model' : model,
      'dataloaders' : cifar_dataloader.get_dataloaders(data_split),
      'optimzer' : optimizer,
      'scheduler' : scheduler,
      'epochs' : epochs,
      'loss_function' : crossentropy_loss,
      'save_config' : config['save_config'],
      'config' : config_summary,  # This dictionary will be saved as json file
      'device' : device_dict
   }
   
   return built_config
   
   
def get_metric_value(metric, device):
   """
   Get value of a metric based on device
   
   Args:
      metric (torch.metric): One of metrics in torch.metric 
      device (str or torch.device): Device
      
   Returns:
      Value of metric
   """
   if device == "cuda" :
      return metric.compute().cuda().item()
   elif device == "cpu" :
      return metric.compute().item()
  

def evaluate(model, test_dataloader, device):
    """
    Evaluate given model on given dataloader and device
    
    Args:
       model(torch.nn.Module): Model to be evaluated
       test_dataloader (torch.data.Dataloader): dataloader of test set
       device (str): Device
       
    Returns:
       Value of evaluation metric
    """
    test_accuracy = MulticlassAccuracy(num_classes=10, device=device)
    
    model = model.to(device)
    
    with torch.no_grad():
        model.eval()
             
        test_batch_data = list(enumerate(test_dataloader))
        tq_bar = tqdm(range(len(test_batch_data)))
        tq_bar.set_description('Evaluation') 
           
        for i in tq_bar :
            test_data, test_label = test_batch_data[i][1][0], test_batch_data[i][1][1]
            test_data, test_label = test_data.to(device), test_label.to(device)
            pred = model(test_data)
            test_accuracy.update(torch.argmax(pred,1), torch.argmax(test_label,1))

    return get_metric_value(test_accuracy, device)

import torch
import math
from torcheval.metrics import MulticlassAccuracy, Mean
from tqdm import tqdm
from .utilities import get_device, sum_squard_weights, get_metric_value, evaluate
from .run_saver import RunSaver

class Trainer:
   """
   Train a model with given model, dataloader, optimizer, scheduler and loss_function
   
   Attributes:
      model (torch.nn.Module): Model to be trained
      dataloader (tuple): Train, validation and test dataloaders in format (Train, Val, Test),
      optimzer (torch.optim.Optimizer): Training optimzer
      scheduler (torch.optim.lr_scheduler.LRScheduler): Training scheduler
      loss_function (function): Function that compute loss value
      
   """
   
   def __init__(self, model, dataloaders, optimizer, scheduler, loss_function) :
      """
      Intialize train
      
      Args:
         model (torch.nn.Module): Model to be trained
         dataloader (tuple): Train, validation and test dataloaders in format (Train, Val, Test),
         optimzer (torch.optim.Optimizer): Training optimzer
         scheduler (torch.optim.lr_scheduler.LRScheduler): Training scheduler
         loss_function (function): Function that compute loss value
      
      """  
      self.model = model
      self.dataloaders = dataloaders # (train, val, test)
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.loss_function = loss_function
      
   def train(self, epochs, save_config, config_summary, devices):
      """
      Train given model with given components
      
      Args:
         epochs (int): Number of epochs to train
         save_config (dict): Dictionary of saving config
         config_summary (dict): Dictionary of train config to be saved as json
         devices (dict): Dictionary of train and evaluation device
         
      """
      # saver config
      saver = RunSaver()
      saver.prepare_new_train_save()
      saver.logger.write_train_info(desc=save_config['log']['desc'])
      log_step = save_config['log']['log_step']
      saver.logger.write_header('Train')
      
      image_save_per_batch = save_config['train_batch_save']['samples']
      batches_to_save = save_config['train_batch_save']['batches_to_save']
      saved_batches = 0
      cols = int(math.sqrt(image_save_per_batch))
      rows = image_save_per_batch // cols
      
      saver.save_config(config_summary)
      
      # detect device
      device = devices['train']
      
      self.model = self.model.to(device)
      saver.save_model_summary(self.model)
      
      # define metrics
      train_accuracy = MulticlassAccuracy(num_classes=10, device=device)
      train_loss_total = Mean(device=device)
      val_accuracy = MulticlassAccuracy(num_classes=10, device=device)
      val_loss = Mean(device=device)

      # best and init
      best_accuracy_train = 0 
      best_accuracy_val = 0
      log_skip = 0
   
      # Train loop
      for epoch in range(epochs) :
         self.model.train()

         batch_data = list(enumerate(self.dataloaders[0]))

         tq_bar = tqdm(range(len(batch_data)))
         tq_bar.set_description('epoch %3d ' % (epoch+1))

         for i in tq_bar :
            if i > 0 :
               tq_bar.set_postfix(ordered_dict={
                 'loss' : get_metric_value(train_loss_total, device),
                 'accuracy' : get_metric_value(train_accuracy, device),
                 'lr': self.scheduler.get_last_lr()[0]})
              
            # write log in every log step
            if log_skip < log_step :
               log_skip += 1
            else:
               saver.logger.write_dictionary({
                        'epoch' : epoch+1,
                        'sum squared weights' : sum_squard_weights(self.model),
                        'loss' : get_metric_value(train_loss_total, device),
                        'accuracy' : get_metric_value(train_accuracy, device),
                        'lr': self.scheduler.get_last_lr()[0]
                    })
               log_skip = 0
                    
    
            data, label = batch_data[i][1][0].to(device), batch_data[i][1][1].to(device)
            
            # save images of batch
            if saved_batches < batches_to_save :
               saved_batches += 1
               saver.save_batch_images(batch_data[i][1][0], cols, rows, saved_batches)
            
            # Compute prediction and loss
            pred = self.model(data)
            pred = torch.nn.functional.softmax(pred, dim=-1)
            cross_entropy_loss = torch.mean(self.loss_function(label, pred))
            loss = cross_entropy_loss
                    
            # update metrics
            train_loss_total.update(loss)
            train_accuracy.update(torch.argmax(pred, 1), torch.argmax(label, 1))
    
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # validation
            if i == len(batch_data)-1 :
                self.model.eval()
                
                with torch.no_grad() :
                    for batch_val, (X_val, y_val) in enumerate(self.dataloaders[1]):
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        
                        # compute validation loss and accuracy
                        pred_val = self.model(X_val)
                        pred_val = torch.nn.functional.softmax(pred_val, dim=-1)
                        loss_val = torch.mean(self.loss_function(y_val, pred_val))
                                
                        # update validation metrics
                        val_loss.update(loss_val)
                        val_accuracy.update(torch.argmax(pred_val, 1), torch.argmax(y_val, 1))
                    tq_bar.set_postfix(ordered_dict={
                        'sum squared weights' : sum_squard_weights(self.model),
                        'loss' : get_metric_value(train_loss_total, device),
                        'accuracy' : get_metric_value(train_accuracy, device),
                        'val loss' : get_metric_value(val_loss, device),
                        'val accuracy' : get_metric_value(val_accuracy, device),
                        'lr': self.scheduler.get_last_lr()[0]})
                    
                    # write log
                    saver.logger.write_dictionary({
                        'epoch' : epoch+1,
                        'sum squared weights' : sum_squard_weights(self.model),
                        'loss' : get_metric_value(train_loss_total, device),
                        'accuracy' : get_metric_value(train_accuracy, device),
                        'val loss' : get_metric_value(val_loss, device),
                        'val accuracy' : get_metric_value(val_accuracy, device),
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                                
         # save best and last
         if train_accuracy.compute() > best_accuracy_train :
             saver.save_model(self.model, 'best_train')
             best_accuracy_train = train_accuracy.compute()
         if val_accuracy.compute() > best_accuracy_val :
             saver.save_model(self.model, 'best_val')
             best_accuracy_val = val_accuracy.compute()
         saver.save_model(self.model, 'last')
         
         # save checkpoint
         state_dict = {
             'epoch' : epoch + 1,
             'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'scheduler_state_dict' : self.scheduler.state_dict()
         }
         saver.save_checkpoint(state_dict)
         
         # scheduler step
         self.scheduler.step()
    
         # reset metrics
         train_accuracy.reset()
         train_loss_total.reset()
         val_accuracy.reset()
         val_loss.reset()
         
      # evaluation
      eval_dict = {}
      saved_models = saver.get_saved_models()
         
      for saved_model in saved_models:
         eval_dict[saved_model[1]] = evaluate(saved_model[0], self.dataloaders[2], devices['eval'])
         
      saver.logger.write_header('Evaluation')
      saver.logger.write_dictionary(eval_dict)
         
    
    
         
import torch
import torchvision
import torchvision.transforms as transforms

class CifarDataloader:
   """
   Prepare CIFAR-10 Datasets and Dataloaders
   """

   def __init__(self, dataset_path, train_batch_size, eval_batch_size, augmentations):

      self.train_batch_size = train_batch_size
      self.eval_batch_size = eval_batch_size
      self.dataset_path = dataset_path
      self.augmentation = augmentations
      
      # normalize transforms array
      self.normalize_transforms = [
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ]
      
      # augmentation and normalize
      self.transform_train = transforms.Compose(self.augmentation + self.normalize_transforms)

      # normalize
      self.transform_test = transforms.Compose(self.normalize_transforms)

      # one-hot transform
      self.one_hot = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

   def get_dataloaders(self, train_split):

      # Train data
      trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=False, transform=self.transform_train, target_transform = self.one_hot)
      train_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(train_split))), batch_size=self.train_batch_size, shuffle=True)

      # Validation data
      valset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=False, transform=self.transform_test, target_transform = self.one_hot)
      val_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(valset, list(range(train_split,50000))), batch_size=self.train_batch_size, shuffle=True)

      # Test data
      testset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=False, transform=self.transform_test, target_transform = self.one_hot)
      test_dataloader = torch.utils.data.DataLoader(testset, batch_size=self.eval_batch_size, shuffle=False)
      
      return train_dataloader, val_dataloader, test_dataloader




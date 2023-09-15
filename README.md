# CIFAR Image Classification

## References

1. https://pytorch.org/tutorials/

## Environment setup

به منظور استفاده و نصب پیشنیاز ها دستور زیر را اجرا کنید :

```
bash ./install.sh
```

## How to use

برای انجام Train ابتدا لازم است دیتاست CIFAR-10 را از لینک زیر  دانلود کنید  و آن را از حالت فشرده خارج نمایید:

https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

برای انجام Train از نمونه فایل config موجود در پوشه configs استفاده کنید و آدرس نسبی دیتاست به فایل `train.py` را در قسمت `dataset_path` قرار دهید. برای انجام Train دستورات زیر را اجرا کنید :

```
source venv/bin/activate
python train.py --config_path {config_file_path}
```


## Evaluation

ارزیابی روی داده های Test دیتاست CIFAR-10 انجام شده است. (دقت گزارش شده برای هر  config میایگین ارزیابی 3 مدل best val از 3 train است)

| Model | Initializer | Activation Function | Agmentations | Optimizer | Scheduler | Batch Size | Epochs | Use Bias | Accuracy |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----- | ------ | ------- |
| Resnet20 | Kaimaing  Uniform | ReLU | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 90.96
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.05) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 90.96 |
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.1) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 91.29 |
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.15) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 91.51 |
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.2) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 91.13 |
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.25) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 91.15 |
| Resnet20 | Kaimaing  Uniform | LeakyReLU(0.3) | pad+hflip+crop | SGD <br/> momentum=0.9 <br/> weight_decay=1e-4 | MultiStepLR <br/> [0.1, 0.01, 0.001] | 128 | 120 | False | 90.74 |


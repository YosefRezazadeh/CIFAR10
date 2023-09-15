import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    """
    Resnet block which repeat in Resnet model architeture
    This class should overwrite '__init__' and 'forward' methods shch as other Pytorch modules
    """

    def __init__(self, ch_in, ch_out, activation_function, use_bias, subsample=False):
        super().__init__()

        self.subsample = subsample

        if subsample :
            self.conv_1 = nn.Conv2d(ch_in, ch_out, 3, 2, padding=1, bias=use_bias)
        else :
            self.conv_1 = nn.Conv2d(ch_in, ch_out, 3, 1, padding=1, bias=use_bias)  

        self.batch_norm_1 = nn.BatchNorm2d(ch_out)
        self.act_1 = activation_function 
        self.conv_2 = nn.Conv2d(ch_out, ch_out, 3, 1, padding=1, bias=use_bias)  
        self.batch_norm_2 = nn.BatchNorm2d(ch_out)

        if subsample :
            self.skip_conv = nn.Conv2d(ch_in, ch_out, 1, 2, bias=use_bias)

        self.act_2 = activation_function 

    def forward(self, x):
        skip = x  # skip connection (residual)

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)

        if self.subsample :
            skip = self.skip_conv(skip)

        logits = skip + x  # adding
        logits = self.act_2(logits)

        return logits
    

class Resnet(nn.Module) :
    """
    Resnet module 
    """

    def __init__(self, architect_list=(3, 3, 3), filter_list=(16, 32, 64), activation_function=nn.ReLU() ,use_bias=False):
        super().__init__()

        self.layer = [
            nn.Conv2d(3, filter_list[0], 3, 1, padding=1, bias=use_bias),
            nn.BatchNorm2d(filter_list[0]),
            activation_function
        ]

        for block_num in range(len(architect_list)):
            for block_layer in range(architect_list[block_num]):
                if block_num > 0 and block_layer == 0 :
                    self.layer.append(ResnetBlock(filter_list[block_num]//2, filter_list[block_num], activation_function, use_bias, True))
                else:
                    self.layer.append(ResnetBlock(filter_list[block_num], filter_list[block_num], activation_function, use_bias))

        self.layer.append(nn.AvgPool2d(8,8))
        self.layer.append(nn.Flatten())
        self.layer.append(nn.Linear(filter_list[-1], 10))
        # self.layer.append(nn.Softmax())

        self.model = nn.Sequential(*self.layer)

    def set_initializer(self, initializer, **kwargs):

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                initializer(m.weight, **kwargs)
  
    def forward(self, x) :

        for l in self.layer:
            x = l(x)

        return x

  
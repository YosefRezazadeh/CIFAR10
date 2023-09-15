import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):

    def __init__(self, ch_in, ch_out, stride, use_bias, expantion_factor=1):
        super().__init__()
    
        ch_expanded = ch_in * expantion_factor

        self.conv = nn.Conv2d(ch_in, ch_expanded, 1, 1, bias=use_bias)
        self.batch_norm_1 = nn.BatchNorm2d(ch_in)
        self.relu_1 = nn.ReLU6()
        self.conv_depth_wise = nn.Conv2d(ch_expanded, ch_expanded, 3, stride, padding=1, bias=use_bias, groups=ch_in*expantion_factor)
        self.batch_norm_2 = nn.BatchNorm2d(ch_in)
        self.relu_2 = nn.ReLU6()
        self.conv_point_wise = nn.Conv2d(ch_expanded, ch_out, 1, 1, 0, bias=use_bias) 
        self.batch_norm_3 = nn.BatchNorm2d(ch_out)
        self.relu_3 = nn.ReLU6()

    def forward(self, x):

        x = self.relu_1(self.batch_norm_1(self.conv(x)))
        x = self.relu_2(self.batch_norm_2(self.conv_depth_wise(x)))
        x = self.relu_3(self.batch_norm_3(self.conv_point_wise(x)))

        return x

class MobilenetV2(nn.Module):

    def __init__(self, architect_list=(4, 6, 4), filter_list=(32, 64, 128), use_bias=False):
        super().__init__()
        
        self.layer = [
            nn.Conv2d(3, filter_list[0], 3, 1, padding=1, bias=use_bias),
            nn.BatchNorm2d(filter_list[0]),
            nn.ReLU()
        ]
        
        for block_num in range(len(architect_list)):
            for block_layer in range(architect_list[block_num]):
                if block_num > 0 and block_layer == 0 :
                    self.layer.append(InvertedResidualBlock(filter_list[block_num]//2, filter_list[block_num], 2, use_bias=use_bias))
                else:
                    self.layer.append(InvertedResidualBlock(filter_list[block_num], filter_list[block_num], 1, use_bias=use_bias))

        self.layer.append(nn.AvgPool2d(7,7))
        self.layer.append(nn.Flatten())
        self.layer.append(nn.Linear(filter_list[-1], 10))
                    
        self.model = nn.Sequential(*self.layer)

    def set_initializer(self, initializer, **kwargs):

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                initializer(m.weight, **kwargs)

    def forward(self, x):
                
        for l in self.layer:
            x = l(x)
            
        return x


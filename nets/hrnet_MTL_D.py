from this import s
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class HRnet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)
        last_inp_channels   = np.int(np.sum(self.backbone.model.pre_stage_channels))
        self.w1             = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        self.w2             = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        self.w1.data.fill_(1)
        self.w2.data.fill_(1)

        self.last_layer_e_cat = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels + num_classes[1], out_channels=last_inp_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes[0], kernel_size=1, stride=1, padding=0)
        )

        self.last_layer_d_cat = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels + num_classes[0], out_channels=last_inp_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes[1], kernel_size=1, stride=1, padding=0)
        )

    def forward(self, inputs, cross_e, cross_d):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x[0], x1, x2, x3], 1)

        x_1  = x * self.w2
        x_1 = torch.concat((x_1, cross_e), 1)
        x_d = self.last_layer_d_cat(x_1)
        x_1 = F.interpolate(x_d, size=(H, W), mode='bilinear', align_corners=True)

        x = x * self.w1
        x = torch.concat((x, cross_d), 1)
        x_e = self.last_layer_e_cat(x)
        x = F.interpolate(x_e, size=(H, W), mode='bilinear', align_corners=True)

        return x, x_1, x_e, x_d

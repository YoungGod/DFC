import torch
import torch.nn as nn
from vgg import vgg19

# BottleNeckDeformableAttention/Conv Module
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = torch.nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = torch.nn.functional.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = torch.nn.functional.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BDCM(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1, kernel_size=3, padding=None, modulation=False, neck_out=False, is_attention=True):
        super(BDCM, self).__init__()
        self.in_channels = in_channels
        self.neck_channels = out_channels//factor
        self.out_channels = out_channels
        self.neck_out = neck_out
        self.factor = factor
        self.is_attention = is_attention

        # bottle neck
        if factor > 1:
            self.neck_head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1)
            if neck_out:
                self.neck_tail = nn.Conv2d(in_channels=self.neck_channels, out_channels=out_channels, kernel_size=1)
        
        # channel attention
        if self.is_attention and factor > 1:
            self.channel_attention = ChannelGate(gate_channels=self.neck_channels, reduction_ratio=16, pool_types=['avg', 'max'])
        if self.is_attention:
            self.channel_attention = ChannelGate(gate_channels=in_channels, reduction_ratio=16, pool_types=['avg', 'max'])

        # deformable conv
        from deform_conv_v2 import DeformConv2d
        if neck_out:
            self.deform_conv = DeformConv2d(inc=self.neck_channels, outc=self.neck_channels, kernel_size=3, padding=None, modulation=True)
        else:
            self.deform_conv = DeformConv2d(inc=self.neck_channels, outc=out_channels, kernel_size=3, padding=None, modulation=False)

    def forward(self, x):
        if self.factor > 1:
            x = self.neck_head(x)
            if self.is_attention:
                x = self.channel_attention(x)
            x = self.deform_conv(x)
            if self.neck_out:
                x = self.neck_tail(x)
        else:
            if self.is_attention:
                x = self.channel_attention(x)
            x = self.deform_conv(x)
        return x

class ResnetBlock_BDCM(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_BDCM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            BDCM(dim, dim, factor=1, kernel_size=3, padding=None, modulation=False, neck_out=False),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
        ) 

    def forward(self, x):
        out = x + self.conv_block(x)    # ReLU or not?
        # print(out.size())   ##########################################
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

# Resblocks
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=2, padding='reflect'):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation) if padding == 'reflect' else nn.ZeroPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1) if padding == 'reflect' else nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
        ) 

    def forward(self, x):
        out = x + self.conv_block(x)    # ReLU or not?
        # print(out.size())   ##########################################
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class MCRL(nn.Module):
    def __init__(self, in_channels=256, residual_blocks=2, dilation=2):
        super(MCRL, self).__init__()

        # deformable branch (data dependent context)
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(in_channels, dilation=2, padding='reflect')
            # block = ResnetBlock_BDCM(in_channels, dilation=dilation)
            blocks.append(block)
        self.middle_res1 = nn.Sequential(*blocks)

        # dilated branch (fixed context)
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(in_channels, dilation=4, padding='reflect')
            blocks.append(block)
        self.middle_res2 = nn.Sequential(*blocks)

        # dilated branch (fixed context)
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(in_channels, dilation=8, padding='reflect')
            blocks.append(block)
        self.middle_res3 = nn.Sequential(*blocks)            

    def forward(self, x):
        out = x + self.middle_res1(x) + self.middle_res2(x) + self.middle_res3(x)
        return out



# Residual Dilated Dense Block
class denseBlockLayer(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize=3, inception = False, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer, self).__init__()
        self.useInception = inception

        if(self.useInception):
            self.conv1 = nn.Conv2d(inChannel,outChannel,3,padding = 1)
            self.conv2 = nn.Conv2d(inChannel,outChannel,5,padding = 2)
            self.conv3 = nn.Conv2d(inChannel,outChannel,7,padding = 3)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            self.conv4 = nn.Conv2d(outChannel*3,outChannel,1,padding = 0)
            #self.relu2 = nn.ReLU()
        else:
            pad = int(dilateScale * (kernelSize - 1) / 2)
            
            self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad, dilation = dilateScale)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            
    def forward(self,x):
        if(self.useInception):
            y2 = x
            y3_1 = self.conv1(y2)
            y3_2 = self.conv1(y2)
            y3_3 = self.conv1(y2)
            y4 = torch.cat((y3_1,y3_2,y3_3),1)
            y4 = self.relu(y4)
            y5 = self.conv4(y4)
            y_ = self.relu(y5)
        else:
            y2 = self.conv(x)
            y_ = self.relu(y2)
            
            
        return y_

class denseBlock(nn.Module):
    def __init__(self, inChannel=64, outChannel=64, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer

        # Global info
        global_out_channel = 64
        self.global_info_layer = GlobalInfo(inChannel=inChannel, outChannel=global_out_channel, kernelSize=3)

        # dense block
        templayerList = []
        # other dense layer
        for i in range(0, layer):
            tempLayer = denseBlockLayer(global_out_channel+inChannel+growthRate*i,growthRate,kernelSize,inceptionLayer,dilate,activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
        self.outputLayer = denseBlockLayer(global_out_channel+inChannel+growthRate*layer,outChannel,kernelSize,inceptionLayer,1,activ)
        # self.bn = nn.BatchNorm2d(outChannel)
        
    def forward(self,x):
        x = self.global_info_layer(x)
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.outputLayer(x)
        # y = self.bn(y)
            
        return y


class GlobalInfo(nn.Module):
    def __init__(self, inChannel=256, outChannel=64, kernelSize=3):
        super(GlobalInfo, self).__init__()
        pad = int((kernelSize-1)/2)
        self.conv = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=kernelSize)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        _, c, h, w = x.shape
        x_g = self.conv(x)
        x_g = self.relu(x_g)
        # avgpool for global information
        x_g = self.pool(x_g)
        x_g = x_g.repeat(1,1,h,w)    # expand spatial dimension
        x = torch.cat([x, x_g], 1)
        return x

class DilatedDenseResidualBlock(nn.Module):
    def __init__(self, in_channels=256, dilated_dense_blocks=2, growthrate=32, num_layer=4, dilation=True):
        super(DilatedDenseResidualBlock, self).__init__()
        self.num_dilated_dense_blocks = dilated_dense_blocks

        dilated_dense_block_list = []
        for i in range(dilated_dense_blocks):
            dilated_dense_block_list.append(denseBlock(inChannel=in_channels, outChannel=in_channels, growthRate=growthrate, layer=num_layer, dilationLayer=True))
        self.dilated_dense_block = nn.ModuleList(dilated_dense_block_list)

    def forward(self, x):
        # residual connection
        for i in range(0, self.num_dilated_dense_blocks):
            x = x + self.dilated_dense_block[i](x)
        return x


class VGG19_S(torch.nn.Module):
    """
    VGG19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    level1: 64*2=128; level2: 128*2=256; level3: 256*4=1024; level4: 512*4=2048; level5: 512*4=2048
    Total dimension: 128 + 256 + 1024 + 2048 + 2048 = 5504
    """
    def __init__(self, pretrain=False, gradient=False, pool='avg'):
        super(VGG19_S, self).__init__()
        features = vgg19(pretrained=pretrain).features    # feature layers
        """ vgg.features
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)                                                        # self.relu1_1
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace)                                                        # self.relu1_2
          
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace)  
              
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace)
          
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace)
          
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (29): ReLU(inplace)
          (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (31): ReLU(inplace)
          (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (33): ReLU(inplace)
          (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (35): ReLU(inplace)
          
          (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        """
        # hierarchy 1 (level 1)
        self.conv1_1 = features[0]
        self.relu1_1 = features[1]
        self.conv1_2 = features[2]
        self.relu1_2 = features[3]

        # hierarchy 2 (level 2)
        self.pool1 = features[4]
        self.conv2_1 = features[5]
        self.relu2_1 = features[6]
        self.conv2_2 = features[7]
        self.relu2_2 = features[8]

        # replace all convs in level 3 with deformable conv v2
        # hierarchy 3 (level 3)    
        self.pool2 = features[9]
        self.conv3_1 = features[10]
        self.relu3_1 = features[11]
        self.conv3_2 = features[12]
        self.relu3_2 = features[13]
        self.conv3_3 = features[14]
        self.relu3_3 = features[15]
        self.conv3_4 = features[16]
        self.relu3_4 = features[17]

        # hierarchy 4 (level 4)
        self.pool3 = features[18]
        self.conv4_1 = features[19]
        self.relu4_1 = features[20]
        self.conv4_2 = features[21]
        self.relu4_2 = features[22]
        self.conv4_3 = features[23]
        self.relu4_3 = features[24]
        self.conv4_4 = features[25]
        self.relu4_4 = features[26]
   
        # after pool3
        self.resnet = MCRL(in_channels=self.conv3_4.in_channels, residual_blocks=4, dilation=2)
        
        # self.resnet = DilatedDenseResidualBlock(in_channels=self.conv3_4.in_channels, dilated_dense_blocks=2, growthrate=32, num_layer=4, dilation=True)
        # # init as zero
        # def weight_init(m):
        #     """
        #     ref: https://blog.csdn.net/qq_36338754/article/details/97756378
        #     """
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         if m.weight is not None:
        #             m.weight.data.zero_()
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        # self.resnet.apply(weight_init)    

        # hierarchy 5 (level 5)
        self.pool4 = features[27]
        self.conv5_1 = features[28]
        self.relu5_1 = features[29]
        self.conv5_2 = features[30]
        self.relu5_2 = features[31]
        self.conv5_3 = features[32]
        self.relu5_3 = features[33]
        self.conv5_4 = features[34]
        self.relu5_4 = features[35]

        self.pool5 = features[36]

        # don't need the gradients, just want the features
        if not gradient:
            for param in self.parameters():
                param.requires_grad = False

        # reflection padding
        self.pad = nn.ReflectionPad2d(padding=1)

        # avg pooling
        if pool == 'avg':
            avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            # self.pool3 = self.pool4 = self.pool5 = avg_pool
            self.pool1 = self.pool2 = self.pool3 = self.pool4 = self.pool5 = avg_pool

    def forward(self, x, feature_layers):
        # level 1    pad before conv
        x = self.pad(x)
        conv1_1 = self.conv1_1(x)
        relu1_1 = self.relu1_1(conv1_1)
        conv1_2 = self.conv1_2(self.pad(relu1_1))
        relu1_2 = self.relu1_2(conv1_2)
        pool1 = self.pool1(relu1_2)

        # level 2
        pool1 = self.pad(pool1)
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = self.relu2_1(conv2_1)
        conv2_2 = self.conv2_2(self.pad(relu2_1))
        relu2_2 = self.relu2_2(conv2_2)
        pool2 = self.pool2(relu2_2)

        # level 3
        pool2 = self.pad(pool2)
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = self.relu3_1(conv3_1)
        conv3_2 = self.conv3_2(self.pad(relu3_1))
        relu3_2 = self.relu3_2(conv3_2)
        conv3_3 = self.conv3_3(self.pad(relu3_2))
        # conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = self.relu3_3(conv3_3)
        conv3_4 = self.conv3_4(self.pad(relu3_3))
        relu3_4 = self.relu3_4(conv3_4)
        pool3 = self.pool3(relu3_4)

        # level 4
        pool3 = self.resnet(pool3)    # resblock
        pool3 = self.pad(pool3)
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = self.relu4_1(conv4_1)
        conv4_2 = self.conv4_2(self.pad(relu4_1))
        relu4_2 = self.relu4_2(conv4_2)
        conv4_3 = self.conv4_3(self.pad(relu4_2))
        relu4_3 = self.relu4_3(conv4_3)
        conv4_4 = self.conv4_4(self.pad(relu4_3))
        relu4_4 = self.relu4_4(conv4_4)
        pool4 = self.pool4(relu4_4)

        # level 5
        pool4 = self.pad(pool4)
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = self.relu5_1(conv5_1)
        conv5_2 = self.conv5_2(self.pad(relu5_1))
        relu5_2 = self.relu5_2(conv5_2)
        # conv5_3 = self.conv5_3(self.pad(relu5_2))
        # relu5_3 = self.relu5_3(conv5_3)
        # conv5_4 = self.conv5_4(self.pad(relu5_3))
        # relu5_4 = self.relu5_4(conv5_4)
        # pool5 = self.pool5(relu5_4)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            # 'relu5_3': relu5_3,
            # 'relu5_4': relu5_4,
        }
        return dict((key, value) for key, value in out.items() if key in feature_layers)

if __name__ == '__main__':
    x = torch.randn((1,3,256,256), dtype=torch.float32)
    model = VGG19_S()
    y = model(x, feature_layers=("relu1_1", "relu1_2", "relu2_1", "relu2_2", 
                                 "relu3_1", "relu3_2", "relu3_3", "relu3_4",
                                 "relu4_1", "relu4_2", "relu4_3", "relu4_4",
                                 "relu5_1", "relu5_2"))    # , "relu5_3", "relu5_4"
    for key, value in y.items():
        print(key, value.shape)
    print(model.conv3_1.in_channels, model.conv3_4.out_channels)
    print(model.conv4_1.in_channels, model.conv4_2.out_channels)
    print(model.conv4_1.kernel_size, model.conv4_2.out_channels)
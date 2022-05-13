import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL_SIZE = (5,3,3)
PAD_SIZE = (2,1,1)

class Attention(nn.Module):

    def __init__(self,bs, c, l=11, h=16, w=16):
        super(Attention,self).__init__()
        self.shape = [bs,c,l,h,w]

        self.conv_1 = nn.Sequential(
            nn.Conv3d(32, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.LeakyReLU(),

            nn.Conv3d(32, 32, KERNEL_SIZE,1,(2,1,1)),
            nn.LeakyReLU(),
        )
        self.avg_poll =  nn.AdaptiveAvgPool3d(
            (11,1,1)
        )


        self.conv_2 = nn.Sequential(
            nn.Linear(11,8,bias=False),
            nn.LeakyReLU(),

            nn.Linear(8,11,bias=False),
            nn.Sigmoid()
        )


    def forward(self,x):

        x1 = self.conv_1(x) 
        x2 = self.avg_poll(x1) 
        x2_1 = torch.squeeze(x2) 
        x3 = self.conv_2(x2_1) 
        x3_1 = x3.reshape(self.shape[0], self.shape[1], self.shape[2], 1, 1) 
        
        x4 = x1 * x3_1

        y = x + x4

        return y



class Generator(nn.Module):


    def __init__(self,bs, c=1, l=11, h=16, w=16):
        super(Generator,self).__init__()
        self.shape = [bs,c,l,h,w]

        self.conv_1 = nn.Sequential(
            nn.Conv3d(1, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.LeakyReLU()
        )

        self.attn_1 = Attention(bs, 32, l, h, w)
        self.attn_2 = Attention(bs, 32, l, h, w)
        self.attn_3 = Attention(bs, 32, l, h, w)

        self.conv_2 = nn.Sequential(
            nn.Conv3d(32, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Conv3d(32, 1, KERNEL_SIZE, 1, (2,1,1))

        self.conv_4 = nn.Sequential(
            nn.Conv3d(1, 1, KERNEL_SIZE, 1, (2,1,1)),
            nn.LeakyReLU()
        )
        
        self.conv_5 = nn.Conv3d(1, 1, KERNEL_SIZE, 1, (2,1,1))

    def forward(self,x):
        
        x1 = self.conv_1(x)

        x2 = self.attn_1(x1)
        x2 = self.attn_2(x2)
        x2 = self.attn_3(x2)

        x2_1 = self.conv_2(x2)

        x3 = x2_1 + x1

        x4 = self.conv_3(x3)
        x4 = torch.squeeze(x4)
        x4 = F.interpolate(x4,scale_factor=1,mode='bicubic')
        x4 = x4.reshape(x4.shape[0],1,x4.shape[1],x4.shape[2],x4.shape[3])
        x4 = self.conv_4(x4)
        

        x4 = torch.squeeze(x4)
        x4 = F.interpolate(x4,scale_factor=2,mode='bicubic')
        x4 = x4.reshape(x4.shape[0],1,x4.shape[1],x4.shape[2],x4.shape[3])
        x4 = self.conv_4(x4)

        x4 = self.conv_4(x4)

        x4 = self.conv_5(x4)

        return x4


class Discriminator(nn.Module):

    def __init__(self,bs, c=1, l=11, h=32, w=32):
        super(Discriminator,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,32,KERNEL_SIZE,1,(2,1,1)),
            nn.LeakyReLU(),

            nn.ConstantPad3d((1,0,1,0,1,2),1),
            nn.Conv3d(32,32,KERNEL_SIZE,2),
            nn.LeakyReLU(),
            nn.Conv3d(32,64,KERNEL_SIZE,1,(2,1,1)),
            nn.LeakyReLU(),
            
            nn.ConstantPad3d((1,0,1,0,1,2),1 ),
            nn.Conv3d(64,64,KERNEL_SIZE,2),
            nn.LeakyReLU(),
            
            nn.Conv3d(64,128,KERNEL_SIZE,1,(2,1,1)),
            nn.LeakyReLU(),
            
            nn.Conv3d(128,128,KERNEL_SIZE,1,(2,1,1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),

        )

        self.linera = nn.Sequential(
            
            nn.Linear(128,256),
            nn.LeakyReLU(),

            nn.Linear(256,1),

        )


    def forward(self,x):

        y = self.conv(x)
        y = torch.squeeze(y)
        y = self.linera(y)


        return y


class ESR_Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(ESR_Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class Spe_loss(nn.Module):
    def __init__(self):
        super(Spe_loss,self).__init__()

    def forward(self,x,y,shape=32):

        if x.dim != 4:
            x = torch.squeeze(x)
            y = torch.squeeze(y)

        loss = 0
        for i in range(shape):
            for j in range(shape):
                fz = (x[:,:,i,j] * y[:,:,i,j]).sum()
                fm = torch.pow((x[:,:,i,j]*x[:,:,i,j]).sum(),0.5) * torch.pow((y[:,:,i,j]*y[:,:,i,j]).sum(),0.5)
                loss += torch.acos(fz/fm)

        return loss / (shape**2)





class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,x,y,l1=1,l2=1e-2,l3=1e-3):

        l1 = nn.L1Loss()
        l1 = l1(x,y)

        ltv = TVLoss()
        ltv = TVLoss(x)
        print(ltv)
        
        ls = l1 + 1e-6*(ltv.data)
        
        le = Spe_loss()
        le = le(x,y)
        
        

        return l1*ls + l2*le 
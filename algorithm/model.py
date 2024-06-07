import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarEncoder(nn.Module):
    def __init__(self, channels_conv):
        super(RadarEncoder, self).__init__()
        self.channels_conv = channels_conv
        self.conv_list = nn.ModuleList()
        self.batchnorm_list = nn.ModuleList()
        
        self.conv_list.append(nn.Conv1d(3,channels_conv[0], kernel_size=1))
        for i in range(len(self.channels_conv)-1):
            self.conv_list.append(nn.Conv1d(channels_conv[i], channels_conv[i+1], kernel_size=1))
            self.batchnorm_list.append(nn.BatchNorm1d(channels_conv[i]))
            
    def forward(self, x):
        for i in range(len(self.channels_conv)-1):
            x = self.batchnorm_list[i](F.relu(self.conv_list[i](x)))
        x = self.conv_list[-1](x)
        x = torch.mean(x, dim=2)
        return x
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , mid_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class RadarCamUNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, radar_channels):
        super(RadarCamUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_mid_channels = len(mid_channels)
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.radar_channels = radar_channels

        self.inc = DoubleConv(in_channels, mid_channels[0])
        for i in range(self.num_mid_channels-2):
            self.down_list.append(Down(mid_channels[i], mid_channels[i+1]))
        
        self.down_list.append(Down(mid_channels[-2], mid_channels[-1]))
        
        self.up_list.append(Up(mid_channels[-1]+radar_channels, mid_channels[-2], mid_channels[-1]+radar_channels-mid_channels[-2]))
        
        for i in range(self.num_mid_channels-2, 1, -1):
            self.up_list.append(Up(mid_channels[i], mid_channels[i-1], mid_channels[i]-mid_channels[i-1]))
        self.up_list.append(Up(mid_channels[1], mid_channels[0], mid_channels[1]-mid_channels[0]))
        
        self.outc = OutConv(mid_channels[0], out_channels)

    def required_radar_size(self, x_depth):
        x_enc = []
        x_enc.append(self.inc(x_depth))
        for i in range(len(self.down_list)):
            x_enc.append(self.down_list[i](x_enc[-1]))
        print("The required radar size is: (batch, " + str(self.radar_channels*x_enc[-1].shape[2]*x_enc[-1].shape[3]) + ")")

    def forward(self, x_depth, x_radar):
        
        x_enc = []
        x_enc.append(self.inc(x_depth))
        for i in range(len(self.down_list)):
            x_enc.append(self.down_list[i](x_enc[-1]))
          
        x_radar = x_radar.view(x_radar.shape[0], self.radar_channels, x_enc[-1].shape[2], x_enc[-1].shape[3])
        x_enc[-1] = torch.cat([x_enc[-1], x_radar], dim=1)
         
        x = self.up_list[0](x_enc.pop(), x_enc.pop())
        
        for i in range(1, len(self.up_list)):
            x = self.up_list[i](x, x_enc.pop())
        x = self.outc(x)
        return x
    
class RadarCamModel(nn.Module):
    def __init__(self, radar_channels_conv, unet_mid_channels, unet_radar_channels):
        super(RadarCamModel, self).__init__()
        self.radar_encoder = RadarEncoder(radar_channels_conv)
        self.unet = RadarCamUNet(1, 1, unet_mid_channels, unet_radar_channels)

    def required_radar_size(self, x_depth):
        self.unet.required_radar_size(x_depth)
        
    def forward(self, depthmap, radar):
        radar_enc = self.radar_encoder(radar)
        output = self.unet(depthmap, radar_enc)
        output = torch.sigmoid(output) # F.softplus(output)
        return output
    
if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    radar_data = torch.randn(32,3,100).to(device)
    depth_data = torch.randn(32,1,450,800).to(device)
    radar_channels_conv =  [64, 128, 256, 432]  # The last channel must match that of model.required_radar_size()
    unet_channels = [2, 4, 8, 12, 16, 24, 32, 48]
    model = RadarCamModel(radar_channels_conv, unet_channels, 24).to(device)
    model.required_radar_size(depth_data)
    output = model(depth_data, radar_data)
    n_params = sum([p.numel() for p in model.parameters()])
    print("Total number of parameters: ", n_params)
    n_params_radar = sum([p.numel() for p in model.radar_encoder.parameters()])
    print("Radar number of parameters: ", n_params_radar)
    n_params_unet = sum([p.numel() for p in model.unet.parameters()])
    print("UNet number of parameters: ", n_params_unet)
    print("Output shape: ", output.shape)
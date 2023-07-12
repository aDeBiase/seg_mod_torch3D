import numpy as np
import torch
from torch import nn
from torch.nn import init
from monai.networks.layers import Act, Norm
from monai.networks.blocks import Convolution

class Conv_layer2D(nn.Module):
    
    def __init__(self, dimensions, in_channels, out_channels, strides, kernel_size=3, act=Act.RELU, norm=Norm.BATCH , is_transposed=False):
        
        super(Conv_layer2D, self).__init__()
        
        self.conv = Convolution(
            dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act = act,
            norm = norm,
            is_transposed=is_transposed,
        )
        
    def forward(self, x):
        
        x = self.conv(x)
        
        return x 

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel, bilSTM):
        super().__init__()
        
        self.bilSTM=bilSTM
        
        if bilSTM:
            self.maxpool=nn.AdaptiveMaxPool2d(1)
            self.avgpool=nn.AdaptiveAvgPool2d(1)
        else:
            self.maxpool=nn.AdaptiveMaxPool3d(1)
            self.avgpool=nn.AdaptiveAvgPool3d(1)
        
        self.se=nn.Sequential(
            Flatten(),
            nn.Linear(in_channel,in_channel//2), #in_channels=in_channel, 
            nn.ReLU(),
            nn.Linear(in_channel//2, out_channel)
        )
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x) :
        
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        
        if self.bilSTM:
            return output.unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            return output.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        

class SpatialAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, bilSTM=False):
        super().__init__()
        
        if bilSTM:
            self.conv=nn.Conv2d(in_channels=in_channel,out_channels=out_channel, kernel_size=kernel_size, stride=1,bias=False) #, padding="same"
        else:
            self.conv=nn.Conv3d(in_channels=in_channel,out_channels=out_channel, kernel_size=kernel_size, stride=1,bias=False) #, padding="same"
        
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        
        output=self.conv(x)
        output=self.sigmoid(output)
        
        return output
    
class SCmodule_block(nn.Module):

    def __init__(self, in_channel, channel, bilSTM):
        
        super().__init__()
        
        self.ca=ChannelAttention(in_channel=in_channel ,out_channel=channel, bilSTM=bilSTM)
        self.sa=SpatialAttention(in_channel=in_channel ,out_channel=channel, kernel_size=1, bilSTM=bilSTM)
        
        self.bilSTM = bilSTM
        
        if bilSTM:
        
            in_channel2=channel
            self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=in_channel2, out_channels=channel, stride=1,kernel_size=1,bias=False),
                nn.ReLU()
            )
            
            self.conv2=nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=channel, stride=1,kernel_size=1,bias=False),
                nn.ReLU()
            )
            
            
        else:
            in_channel2=in_channel*2
        
            self.conv1=nn.Sequential(
                nn.Conv3d(in_channels=in_channel2, out_channels=channel, stride=1,kernel_size=1,bias=False),
                nn.ReLU()
            )
            
            self.conv2=nn.Sequential(
                nn.Conv3d(in_channels=in_channel, out_channels=channel, stride=1,kernel_size=1,bias=False),
                nn.ReLU()
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                  
    def forward(self, x, y):
        
        x_ = self.conv1(x)
        y_ = self.conv2(y)
        
        F = x_+y_
        
        x_new = x_ + F*self.ca(F)
        y_new = y_ + F*self.sa(F)
        
        return [x_new,y_new]
    
    
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan, out_C):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.scblock_1 = SCmodule_block(nf, nf, bilSTM=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.scblock_2 = SCmodule_block(nf, nf, bilSTM=True)
        
        self.encoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.scblock_3 = SCmodule_block(nf, nf, bilSTM=True)
        
   
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.decoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=out_C,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_seq, h_t, c_t):

        # encoder
        
        h_t1, c_t1 = self.encoder_1_convlstm(input_tensor=x[:, 0, :, :, :], cur_state=[h_t, c_t])  # we could concat to provide skip conn here
        
        xCA, hSA = self.scblock_1(x[:, 0, :, :, :], h_t1)
        
        h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=x[:, 1, :, :, :], cur_state=[hSA, c_t1])  # we could concat to provide skip conn here
        xCA2, hSA2 = self.scblock_2(x[:, 1, :, :, :], h_t2)
        
        
        h_t3, c_t3 = self.encoder_3_convlstm(input_tensor=x[:, 2, :, :, :], cur_state=[hSA2, c_t2])  # we could concat to provide skip conn here
        xCA3, hSA3 = self.scblock_3(x[:, 2, :, :, :], h_t3)
        
        ############
        
        # decoder
        
        h1, co_1 = self.decoder_1_convlstm(input_tensor=xCA3, cur_state=[hSA3, c_t3])
        
        h2, co_2 = self.decoder_2_convlstm(input_tensor=xCA2, cur_state=[h1, co_1])
        
        h3, co_3 = self.decoder_3_convlstm(input_tensor=xCA, cur_state=[h2, co_2])
        
        ############
        
        outputs = [h3, h2, h1]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Softmax()(outputs)
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs
    

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t)

        return outputs

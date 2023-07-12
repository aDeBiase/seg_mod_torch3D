import torch
from torch import nn
from monai.networks.layers.factories import Act
from monai.networks.blocks import UpSample, MaxAvgPool
from segmentation.ops_lynph import Conv_layer2D, SCmodule_block, EncoderDecoderConvLSTM

class Generator_VGG16_mini_SE(nn.Module):
    
    def __init__(self, dims=2, in_channels=2, out_channels=3, gf_dim=64, kernel_size=3):
        
        super(Generator_VGG16_mini_SE, self).__init__()
        
        self.block_1_1_left = Conv_layer2D(dims, in_channels, gf_dim, strides=1)
        self.block_1_2_left = Conv_layer2D(dims, gf_dim, gf_dim, strides=1)  #---> concatenate
        
        self.maxPool = MaxAvgPool(dims,kernel_size=[2,2,1])
        
        self.block_2_1_left = Conv_layer2D(dims, gf_dim*2, gf_dim*2, strides=1)
        self.block_2_2_left = Conv_layer2D(dims, gf_dim*2, gf_dim*2, strides=1) #---> concatenate
        
        self.block_3_1_left = Conv_layer2D(dims, gf_dim*4, gf_dim*4, strides=1)
        self.block_3_2_left = Conv_layer2D(dims, gf_dim*4, gf_dim*4, strides=1)
        self.block_3_3_left = Conv_layer2D(dims, gf_dim*4, gf_dim*4, strides=1) #---> concatenate
        
        self.block_4_1_left = Conv_layer2D(dims, gf_dim*8, gf_dim*8, strides=1)
        self.block_4_2_left = Conv_layer2D(dims, gf_dim*8, gf_dim*8, strides=1)
        self.block_4_3_left = Conv_layer2D(dims, gf_dim*8, gf_dim*8, strides=1) #---> concatenate
        
        self.block_3_1_right = Conv_layer2D(dims, gf_dim*8, gf_dim*8, strides=1, is_transposed=True)
        self.block_3_2_right = Conv_layer2D(dims, gf_dim*8, gf_dim*8, strides=1, is_transposed=True)
        
        self.upSamp1 = UpSample(dims,in_channels=gf_dim*8,scale_factor=[2,2,1])
        
        self.scblock_3 = SCmodule_block(gf_dim*4, gf_dim*4, bilSTM=False)
        
        self.block_2_1_right = Conv_layer2D(dims, gf_dim*8, gf_dim*4, strides=1, is_transposed=True)
        self.block_2_2_right = Conv_layer2D(dims, gf_dim*4, gf_dim*4, strides=1, is_transposed=True) 
        
        self.upSamp2 = UpSample(dims,in_channels=gf_dim*4,scale_factor=[2,2,1])
        
        self.scblock_2 = SCmodule_block(gf_dim*2, gf_dim*2, bilSTM=False)
        
        self.block_1_1_right = Conv_layer2D(dims, gf_dim*4, gf_dim*2, strides=1, is_transposed=True)
        self.block_1_2_right = Conv_layer2D(dims, gf_dim*2, gf_dim*2, strides=1, is_transposed=True)
        
        self.upSamp3 = UpSample(dims,in_channels=gf_dim*2,scale_factor=[2,2,1])
        
        self.scblock_1 = SCmodule_block(gf_dim, gf_dim, bilSTM=False)
        
        self.output_layer = Conv_layer2D(dims, gf_dim*2, out_channels, strides=1, act="softmax", is_transposed=True) 
        
    def forward(self, x):
        
        en0 = self.block_1_2_left(self.block_1_1_left(x))
        en0_ = self.maxPool(en0)
        en1 = self.block_2_2_left(self.block_2_1_left(en0_))
        en1_ = self.maxPool(en1)
        en2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(en1_)))
        en2_ = self.maxPool(en2) #[1, 512, 18, 54]
        en3 = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(en2_)))
        
        de0 = self.block_3_2_right(self.block_3_1_right(en3)) #[1, 512, 18, 54]
        de0 = self.upSamp1(de0) #[1, 512, 36, 108]
        de0_sc = torch.cat(self.scblock_3(de0, en2),1)
        de1 = self.block_2_2_right(self.block_2_1_right(de0_sc))
        de1 = self.upSamp2(de1)
        de1_sc = torch.cat(self.scblock_2(de1, en1),1)
        de2 = self.block_1_2_right(self.block_1_1_right(de1_sc))
        de2 = self.upSamp3(de2)
        de2_sc = torch.cat(self.scblock_1(de2, en0),1)
        
        return [self.output_layer(de2_sc), de2_sc]


class Generator_middle(nn.Module):
    
    def __init__(self, dims=3, in_channels=128, out_channels=3, gf_dim=64, kernel_size=3):
        
        super().__init__()
        
        self.block = nn.Sequential(
                              Conv_layer2D(dims, in_channels, gf_dim, strides=1, kernel_size=3, is_transposed=True),
                              Conv_layer2D(dims, gf_dim, gf_dim, strides=1, kernel_size=3, is_transposed=True))
        
        self.output_layer = Conv_layer2D(dims, gf_dim, out_channels, strides=1, act="softmax", is_transposed=True) 
        
    def forward(self, x):
        
        x = self.block(x)
        
        return [self.output_layer(x), x]


generators={"VGG16_mini_SE" : Generator_VGG16_mini_SE}


class BILSTM_SC_UNET(nn.Module):
    
    def __init__(self, dims=3, in_channels=2, out_channels=3, kernel_size=3, pool_kernel=2, act="relu", generator_model=generators["VGG16_mini_SE"]):
        
        super(BILSTM_SC_UNET, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = generator_model(dims=3, in_channels=2, out_channels=3, gf_dim=64, kernel_size=3).to(device)
        self.middle_generator = Generator_middle(dims=3, in_channels=128, out_channels=3, gf_dim=64, kernel_size=3).to(device)
        self.BiLSTM = EncoderDecoderConvLSTM(nf=64, in_chan=64, out_C=3).to(device)
  
    def forward(self, x):
        
        first_output, x = self.generator(x)
        second_output, x = self.middle_generator(x)
        
        #_, _, h, w = x.size()
        
        #im_size = int(w/3)
        
        x_new = (x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2])
        x_new = torch.stack(x_new, 1)
        #print(x_new.shape)
        third_output = self.BiLSTM(x_new)
        #print(third_output.shape)
        third_output = third_output.permute(0, 2, 3, 4, 1)
        
        """
        Parameters -> reorganize!
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        
        return [first_output, second_output, third_output]
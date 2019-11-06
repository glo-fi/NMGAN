import argparse
import os
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.utils.spectral_norm as spectral_norm
import torch.utils.data as data
import numpy as np

train_folder = r'data/sonatas_8s'
lq_train_folder = r'data/sonatas_lq_8s'
hq_test_folder = r'data/sonatas_minitest_8s'
lq_test_folder = r'data/sonatas_lq_minitest_8s'
serialized_train_folder = r'data/sonatas_serial_8s'
serialized_test_folder = r'data/sonatas_serial_minitest_8s'



class SonataDataset(data.Dataset):

    """

    Audio sample reader for our Sonata Dataset.

    """



    def __init__(self, data_type):
        if data_type == 'train':
            data_path = serialized_train_folder
        else:
            data_path = serialized_test_folder
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))



        self.data_type = data_type

        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]



    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        lq = pair[1].reshape(1, -1)
        if self.data_type == 'train':
            hq = pair[0].reshape(1, -1)
            return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(hq).type(torch.FloatTensor), torch.from_numpy(lq).type(torch.FloatTensor)
        else:
            return os.path.basename(self.file_names[idx]), torch.from_numpy(lq).type(torch.FloatTensor)


    def __len__(self):
        return len(self.file_names)


class Discriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super().__init__()
        # Define convolution operations.
        # (#input channel, #output channel, kernel_size, stride, padding)
        # in : 128,000 x 1
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)   # out : 32000 x 32
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = spectral_norm(nn.Conv1d(32, 64, 31, 2, 15))  # 16000 x 64
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = spectral_norm(nn.Conv1d(64, 64, 31, 2, 15))  # 8000 x 64
        self.dropout1 = nn.Dropout(dropout_drop)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = spectral_norm(nn.Conv1d(64, 128, 31, 2, 15))  # 4000 x 128
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = spectral_norm(nn.Conv1d(128, 128, 31, 2, 15))  # 2000 x 128
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = spectral_norm(nn.Conv1d(128, 256, 31, 2, 15))  # 1000 x 256
        self.dropout2 = nn.Dropout(dropout_drop)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = spectral_norm(nn.Conv1d(256, 256, 31, 2, 15))  # 500 x 256
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = spectral_norm(nn.Conv1d(256, 512, 31, 2, 15))  # 250 x 512
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = spectral_norm(nn.Conv1d(512, 512, 31, 2, 15)) # 125 x 1024
        self.dropout3 = nn.Dropout(dropout_drop)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = spectral_norm(nn.Conv1d(512, 1024, 29, 2, 15)) # 64 x 2048
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = spectral_norm(nn.Conv1d(1024, 1024, 31, 2, 15)) 
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        #self.conv12 = spectral_norm(nn.Conv1d(1024, 2048, 31, 2, 15))
        #self.lrelu12 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = nn.Conv1d(1024, 1, kernel_size=1, stride=1)  # 1024 x 1
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=32, out_features=1)  # 1
        self.sigmoid = nn.Sigmoid()
        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        """
        Forward pass of discriminator.
        Args:
            x: batch
        """
        # train pass
        x = self.conv1(x)
        x = self.lrelu1(x)
        
        x = self.conv2(x)
        x = self.lrelu2(x)
        
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.lrelu3(x)
        
        x = self.conv4(x)
        x = self.lrelu4(x)
        
        x = self.conv5(x)
        x = self.lrelu5(x)
        
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.lrelu6(x)
        
        x = self.conv7(x)
        x = self.lrelu7(x)
        
        x = self.conv8(x)
        x = self.lrelu8(x)

        x = self.conv9(x)
        x = self.dropout3(x)
        x = self.lrelu9(x)

        x = self.conv10(x)
        x = self.lrelu10(x)

        x = self.conv11(x)
        x = self.lrelu11(x)

        #x = self.conv12(x)
        #x = self.lrelu12(x)	

        x = self.conv_final(x)
        x = self.lrelu_final(x)
        
        # reduce down to a scalar value
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)


class Generator(nn.Module):

    """G"""



    def __init__(self):

        super().__init__()
        negative_slope = 0.03

        # encoder gets a lq signal as input [B x 1 x 64000]

        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)  # [B x 16 x 32000]
        self.bn1 = nn.BatchNorm1d(16)
        self.enc1_nl = nn.PReLU()

        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # [B x 32 x 16000]
        self.bn2 = nn.BatchNorm1d(32)
        self.enc2_nl = nn.PReLU()

        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # [B x 32 x 8000] 
        self.bn3 = nn.BatchNorm1d(32)
        self.enc3_nl = nn.PReLU()
        

        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # [B x 64 x 4000]
        self.bn4 = nn.BatchNorm1d(64)
        self.enc4_nl = nn.PReLU()

        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # [B x 64 x 2000]
        self.bn5 = nn.BatchNorm1d(64)
        self.enc5_nl = nn.PReLU()

        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # [B x 128 x 1000]
        self.bn6 = nn.BatchNorm1d(128)
        self.enc6_nl = nn.PReLU()

        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # [B x 128 x 500]
        self.bn7 = nn.BatchNorm1d(128)
        self.enc7_nl = nn.PReLU()

        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # [B x 256 x 250]
        self.bn8 = nn.BatchNorm1d(256)
        self.enc8_nl = nn.PReLU()

        self.enc9 = nn.Conv1d(256, 256, 32, 2, 15) # [B x 256 x 125]
        self.bn9 = nn.BatchNorm1d(256)
        self.enc9_nl = nn.PReLU()

        self.enc10 = nn.Conv1d(256, 512, 29, 2, 15) # [B x 512 x 64]
        self.bn10 = nn.BatchNorm1d(512)
        self.enc10_nl = nn.PReLU()

        self.enc11 = nn.Conv1d(512, 512, 32, 2, 15)  # [B x 1024 x 32]
        self.bn11 = nn.BatchNorm1d(512)
        self.enc11_nl = nn.PReLU()
        
        self.enc12 = nn.Conv1d(512, 1024, 32, 2, 15)  # [B x 1024 x 16]
        self.bn12 = nn.BatchNorm1d(1024)
        self.enc12_nl = nn.Tanh()

        #self.enc13 = nn.Conv1d(1024, 1024, 32, 2, 15)  # [B x 2048 x 8]
        #self.bn13 = nn.BatchNorm1d(1024)
        #self.enc13_nl = nn.PReLU()
        
       # self.enc14 = nn.Conv1d(1024, 2048, 32, 2, 15)  # [B x 2048 x 8]
       # self.bn14 = nn.BatchNorm1d(2048)
       # self.enc14_nl = nn.Tanh()

        # decoder generatean enhanced signal

        # each decoder output are concatenated with homologous encoder output,

        # so the feature map sizes are doubled

      #  self.dec13 = nn.ConvTranspose1d(4096, 1024, 32, 2, 15)  # [B x 2048 x 16]
      #  self.dec_bn13 = nn.BatchNorm1d(1024)
      #  self.dec13_nl = nn.LeakyReLU(negative_slope)
        
        #self.dec12 = nn.ConvTranspose1d(2048, 1024, 32, 2, 15)  # [B x 1024 x 16]
        #self.dec_bn12 = nn.BatchNorm1d(1024)
        #self.dec12_nl = nn.LeakyReLU(negative_slope)
        
        self.dec11 = nn.ConvTranspose1d(2048, 512, 32, 2, 15)  # [B x 512 x 16]
        self.dec_bn11 = nn.BatchNorm1d(512)
        self.dec11_nl = nn.LeakyReLU(negative_slope)
        
        self.dec10 = nn.ConvTranspose1d(1024, 512, 32, 2, 15)  # [B x 512 x 32]
        self.dec_bn10 = nn.BatchNorm1d(512)
        self.dec10_nl = nn.LeakyReLU(negative_slope)

        self.dec9 = nn.ConvTranspose1d(1024, 256, 29, 2, 15)
        self.dec_bn9 = nn.BatchNorm1d(256)
        self.dec9_nl = nn.LeakyReLU(negative_slope)

        self.dec8 = nn.ConvTranspose1d(512, 256, 32, 2, 15)
        self.dec_bn8 = nn.BatchNorm1d(256)
        self.dec8_nl = nn.LeakyReLU(negative_slope)

        self.dec7 = nn.ConvTranspose1d(512, 128, 32, 2, 15)  # [B x 128 x 250]
        self.dec_bn7 = nn.BatchNorm1d(128)
        self.dec7_nl = nn.LeakyReLU(negative_slope)

        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # [B x 128 x 500]
        self.dec_bn6 = nn.BatchNorm1d(128)
        self.dec6_nl = nn.LeakyReLU(negative_slope)

        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # [B x 64 x 1000]
        self.dec_bn5 = nn.BatchNorm1d(64)
        self.dec5_nl = nn.LeakyReLU(negative_slope)

        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # [B x 64 x 2000]
        self.dec_bn4 = nn.BatchNorm1d(64)
        self.dec4_nl = nn.LeakyReLU(negative_slope)

        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # [B x 32 x 4000]
        self.dec_bn3 = nn.BatchNorm1d(32)
        self.dec3_nl = nn.LeakyReLU(negative_slope)

        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # [B x 32 x 8000]
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.dec2_nl = nn.LeakyReLU(negative_slope)

        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # [B x 16 x 16000]
        self.dec_bn1 = nn.BatchNorm1d(16)
        self.dec1_nl = nn.LeakyReLU(negative_slope)

        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # [B x 1 x 32000]
        
        self.dec_tanh = nn.Tanh()



        # initialize weights

        self.init_weights()



    def init_weights(self):

        """

        Initialize weights for convolution layers using Xavier initialization.

        """

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)



    def forward(self, x, z):

        """

        Forward pass of generator.



        Args:

            x: input batch (signal)

            z: latent vector

        """

        # encoding step

        e1 = self.enc1(x)
        b1 = self.bn1(e1)
        e2 = self.enc2(self.enc1_nl(b1))
        b2 = self.bn2(e2)
        e3 = self.enc3(self.enc2_nl(b2))
        b3 = self.bn3(e3)
        e4 = self.enc4(self.enc3_nl(b3))
        b4 = self.bn4(e4)
        e5 = self.enc5(self.enc4_nl(b4))
        b5 = self.bn5(e5)
        e6 = self.enc6(self.enc5_nl(b5))
        b6 = self.bn6(e6)
        e7 = self.enc7(self.enc6_nl(b6))
        b7 = self.bn7(e7)
        e8 = self.enc8(self.enc7_nl(b7))
        b8 = self.bn8(e8)
        e9 = self.enc9(self.enc8_nl(b8))
        b9 = self.bn9(e9)
        e10 = self.enc10(self.enc9_nl(b9))
        b10 = self.bn10(e10)
        e11 = self.enc11(self.enc10_nl(b10))
        b11 = self.bn11(e11)
        e12 = self.enc12(self.enc11_nl(b11))
        b12 = self.bn12(e12)
        #e13 = self.enc13(self.enc12_nl(b12))
        #b13 = self.bn13(e13)
       # e14 = self.enc14(self.enc13_nl(b13))
       # b14 = self.bn14(e14)

        # c = compressed feature, the 'thought vector'
        c = self.enc12_nl(b12)

        # concatenate the thought vector with latent variable

        encoded = torch.cat((c, z), dim=1)

        
        # decoding step        e11 = self.enc11(self.enc10_nl(b10))
       # d13 = self.dec13(encoded)
       # dec_b13 = self.dec_bn13(d13)
       # d13_c = self.dec13_nl(torch.cat((dec_b13, e13), dim=1))
        #d12 = self.dec12(encoded)
        #dec_b12 = self.dec_bn12(d12)
        #d12_c = self.dec12_nl(torch.cat((dec_b12, e12), dim=1))
        d11 = self.dec11(encoded)
        dec_b11 = self.dec_bn11(d11)
        d11_c = self.dec11_nl(torch.cat((dec_b11, e11), dim=1))
        d10 = self.dec10(d11_c)
        dec_b10 = self.dec_bn10(d10)
        d10_c = self.dec10_nl(torch.cat((dec_b10, e10), dim=1))
        d9 = self.dec9(d10_c)
        dec_b9 = self.dec_bn9(d9)
        d9_c = self.dec9_nl(torch.cat((dec_b9, e9), dim=1))
        d8 = self.dec8(d9_c)
        dec_b8 = self.dec_bn8(d8)
        d8_c = self.dec8_nl(torch.cat((dec_b8, e8), dim=1))
        d7 = self.dec7(d8_c)
        dec_b7 = self.dec_bn7(d7)
        d7_c = self.dec7_nl(torch.cat((dec_b7, e7), dim=1))
        d6 = self.dec6(d7_c)
        dec_b6 = self.dec_bn6(d6)
        d6_c = self.dec6_nl(torch.cat((dec_b6, e6), dim=1))
        d5 = self.dec5(d6_c)
        dec_b5 = self.dec_bn5(d5)
        d5_c = self.dec5_nl(torch.cat((dec_b5, e5), dim=1))
        d4 = self.dec4(d5_c)
        dec_b4 = self.dec_bn4(d4)
        d4_c = self.dec4_nl(torch.cat((dec_b4, e4), dim=1))
        d3 = self.dec3(d4_c)
        dec_b3 = self.dec_bn3(d3)
        d3_c = self.dec3_nl(torch.cat((dec_b3, e3), dim=1))
        d2 = self.dec2(d3_c)
        dec_b2 = self.dec_bn2(d2)
        d2_c = self.dec2_nl(torch.cat((dec_b2, e2), dim=1))
        d1 = self.dec1(d2_c)
        dec_b1 = self.dec_bn1(d1)
        d1_c = self.dec1_nl(torch.cat((dec_b1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out

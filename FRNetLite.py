#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import time


def compress(x, beta=0.5):
    
    # (B 2*F T)
    x_real, x_imag = torch.chunk(x, 2, dim=1)
    x_mag = ((x_real**2+x_imag**2)**0.5)**beta
    x_phase = torch.atan2(x_imag, x_real)
    
    real = x_mag*torch.cos(x_phase)
    imag = x_mag*torch.sin(x_phase)
    
    out = torch.cat([real, imag], dim=1)
    
    return out

def stft(x, win_len, win_inc):
    
    X = torch.stft(x, win_len, win_inc, win_len, torch.hann_window(win_len, device=x.device), return_complex=False)
    X_real, X_imag = torch.chunk(X, 2, dim=-1)
    Y = torch.cat([X_real.squeeze(-1), X_imag.squeeze(-1)], dim=1)
    
    return Y

def istft(x, win_len, win_inc):
    
    x_real, x_imag = torch.chunk(x, 2, dim=1)
    X = torch.stack([x_real, x_imag], dim=-1)
    y = torch.istft(X, win_len, win_inc, win_len, torch.hann_window(win_len, device=x.device), return_complex=False)

    return y

# https://github.com/Andong-Li-speech/TaylorSENet/blob/main/utils/utils.py
class CumulativeLayerNorm1d(nn.Module):
    
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B C T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B T)

        entry_cnt = torch.arange(channel, channel*seq_len+1, channel, dtype=inpt.dtype, device=inpt.device)
        entry_cnt = entry_cnt.view(1, -1)  # (B T)

        cum_mean = cum_sum/entry_cnt  # (B T)
        cum_var = (cum_power_sum-2*cum_mean*cum_sum)/entry_cnt+cum_mean.pow(2)
        cum_std = (cum_var+self.eps).sqrt()

        x = (inpt-cum_mean.unsqueeze(dim=1))/cum_std.unsqueeze(dim=1)
        
        return x*self.gain+self.bias

class CumulativeLayerNorm2d(nn.Module):
    
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features[0], 1, num_features[1]), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features[0], 1, num_features[1]), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features[0], 1, num_features[1]), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features[0], 1, num_features[1]), requires_grad=False)

    def forward(self, inpt):
        # inpt: (B C T F)
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1, 3], keepdim=True)  # (B 1 T 1)
        step_pow_sum = inpt.pow(2).sum([1, 3], keepdim=True)  # (B 1 T 1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B 1 T 1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B 1 T 1)

        entry_cnt = torch.arange(channel*freq_num, channel*freq_num*seq_len+1, channel*freq_num, dtype=inpt.dtype, device=inpt.device)
        entry_cnt = entry_cnt.view(1, 1, seq_len, 1)

        cum_mean = cum_sum/entry_cnt
        cum_var = (cum_pow_sum-2*cum_mean*cum_sum)/entry_cnt+cum_mean.pow(2)
        cum_std = (cum_var+self.eps).sqrt()

        x = (inpt-cum_mean)/cum_std
        
        return x*self.gain+self.bias

class PReLU1d(nn.Module):
    
    def __init__(self, num_features):
        super(PReLU1d, self).__init__()
        
        self.a = nn.Parameter(0.25*torch.ones(1, num_features, 1), requires_grad=True)
        
    def forward(self, x):
        
        return torch.clamp(x, 0, None)+self.a*torch.clamp(x, None, 0)
    
class PReLU2d(nn.Module):
    
    def __init__(self, num_features):
        super(PReLU2d, self).__init__()
        
        self.a = nn.Parameter(0.25*torch.ones(1, num_features[0], 1, num_features[1]), requires_grad=True)
        
    def forward(self, x):
        
        return torch.clamp(x, 0, None)+self.a*torch.clamp(x, None, 0)

class Pad1d(nn.Module):
    
    def __init__(self, pad_size):
        super(Pad1d, self).__init__()
        
        self.pad_size = pad_size
        
    def forward(self, x):
        
        return nn.functional.pad(x, [self.pad_size, 0]).contiguous()

class Pad2d(nn.Module):
    
    def __init__(self, pad_size):
        super(Pad2d, self).__init__()
        
        self.pad_size = pad_size
        
    def forward(self, x):
        
        return nn.functional.pad(x, [0, 0, self.pad_size, 0]).contiguous()

class Chomp1d(nn.Module):
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        
        self.chomp_size = chomp_size

    def forward(self, x):
        
        if self.chomp_size != 0:
            return x[:, :, :-self.chomp_size].contiguous()
        else:
            return x
    
class Chomp2d(nn.Module):
    
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        
        self.chomp_size = chomp_size

    def forward(self, x):
        
        if self.chomp_size != 0:
            return x[:, :, :-self.chomp_size, :].contiguous()
        else:
            return x

def CRM(S_real, S_imag, G, F_real, F_imag):
    
    # (B F T)
    S_mag = (S_real**2+S_imag**2)**0.5
    S_phase = torch.atan2(S_imag, S_real)
    
    S_mag_f = S_mag*G
    S_real_f = S_mag_f*torch.cos(S_phase)+F_real
    S_imag_f = S_mag_f*torch.sin(S_phase)+F_imag
    
    return S_real_f, S_imag_f

class GLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, feature_size):
        super(GLUConv2d, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.pad = Pad2d(kernel_size[0]-stride[0])
        self.sigmoid = nn.Sigmoid()
        self.norm = CumulativeLayerNorm2d([out_channels, feature_size])
        self.relu = PReLU2d([out_channels, feature_size])

    def forward(self, x):
        
        out1 = self.conv1(self.pad(x))
        out2 = self.conv2(self.pad(x))
        out = self.relu(self.norm(out1*self.sigmoid(out2)))
        
        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, feature_size):
        super(Conv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.pad = Pad2d(kernel_size[0]-stride[0])
        self.norm = CumulativeLayerNorm2d([out_channels, feature_size])
        self.relu = PReLU2d([out_channels, feature_size])

    def forward(self, x):
        
        out = self.relu(self.norm(self.conv(self.pad(x))))
        
        return out

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, feature_size, mode):
        super(ConvTranspose2d, self).__init__()
        
        if mode == 'cat':
            self.convt = nn.ConvTranspose2d(in_channels=2*in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride)
        if mode == 'add':
            self.convt = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride)
        self.chomp = Chomp2d(kernel_size[0]-stride[0])
        self.norm = CumulativeLayerNorm2d([out_channels, feature_size])
        self.relu = PReLU2d([out_channels, feature_size])

    def forward(self, x):
        
        out = self.relu(self.norm(self.chomp(self.convt(x))))
        
        return out

class GLUTCMBlock(nn.Module):
    
    def __init__(self, dilation, hidden_size=64):
        super(GLUTCMBlock, self).__init__()
        
        self.pad = Pad1d(dilation*(3-1))
        self.relu1 = PReLU1d(hidden_size)
        self.norm1 = CumulativeLayerNorm1d(hidden_size)
        self.convD1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, dilation=dilation)
        self.relu2 = PReLU1d(hidden_size)
        self.norm2 = CumulativeLayerNorm1d(hidden_size)
        self.convD2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, dilation=dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):        
        
        path1 = nn.Sequential(self.relu1,
                              self.norm1,
                              self.pad,
                              self.convD1)
        path2 = nn.Sequential(self.relu2,
                              self.norm2,
                              self.pad,
                              self.convD2)
        out1 = path1(x)
        out2 = path2(x)
        out = out1*self.sigmoid(out2)
        
        return out
    
class STCMBlock(nn.Module):
    
    def __init__(self, dilation, channels=256, hidden_size=64):
        super(STCMBlock, self).__init__()
        
        self.conv_in = nn.Conv1d(in_channels=channels, out_channels=hidden_size, kernel_size=1)
        self.glu = GLUTCMBlock(dilation, hidden_size)
        self.relu = PReLU1d(hidden_size)
        self.norm = CumulativeLayerNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(in_channels=hidden_size, out_channels=channels, kernel_size=1)
        
    def forward(self, x):
        
        net = nn.Sequential(self.conv_in,
                            self.glu,
                            self.relu,
                            self.norm,
                            self.conv_out)
        out = net(x)+x
        
        return out
    
class STCM(nn.Module):
    
    def __init__(self, channels=256, hidden_size=64, dilations=(1, 2, 5, 9), num_repeats=2):
        super(STCM, self).__init__()
        
        self.stcm = self._stcm(channels, hidden_size, dilations, num_repeats)
        
    def forward(self, x):
        
        return self.stcm(x)
        
    def _block(self, channels=256, hidden_size=64, dilations=(1, 2, 5, 9)):
    
        block_list = [STCMBlock(i, channels, hidden_size) for i in dilations]
        return nn.Sequential(*block_list)
    
    def _stcm(self, channels=256, hidden_size=64, dilations=(1, 2, 5, 9), num_repeats=2):
        
        stcm_list = [self._block(channels, hidden_size, dilations) for _ in range(num_repeats)]
        return nn.Sequential(*stcm_list)

class U2NetModule(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 kernel_size_path,
                 stride_path,
                 channel_list,
                 feature_list,
                 num_blocks,
                 dilation):
        super(U2NetModule, self).__init__()
        
        self.glu_in = GLUConv2d(in_channels, out_channels, kernel_size, stride_path, feature_list[-num_blocks-1])
        self.kernel_size = kernel_size_path
        self.stride = stride_path
        self.feature_list = feature_list
        
        encoder_list = self.UNetBlock_en(channel_list)
        decoder_list = self.UNetBlock_de(channel_list)
        
        self.encoder = nn.ModuleList(encoder_list[-num_blocks:])
        self.decoder = nn.ModuleList(decoder_list[:num_blocks])
        self.path = nn.Sequential(Pad2d(dilation*(3-1)),
                                  nn.Conv2d(in_channels=channel_list[-1], 
                                            out_channels=channel_list[-1], 
                                            kernel_size=(3, 1),
                                            stride=(1, 1),
                                            dilation=(dilation, 1)),
                                  CumulativeLayerNorm2d([channel_list[-1], feature_list[-1]]),
                                  PReLU2d([channel_list[-1], feature_list[-1]]))
        
    def forward(self, x):
        
        x1 = self.glu_in(x)
        x = x1
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)
        x = self.path(x)
        for i in range(len(self.decoder)):
            x_cat = torch.cat([x, x_list[-(i+1)]], dim=1)
            x = self.decoder[i](x_cat)
       
        x = x+x1
        del x_list
        
        return x
    
    def UNetBlock_en(self, channel_list):
        
        return [Conv2d(channel_list[1], channel_list[1], self.kernel_size, self.stride, self.feature_list[1]),
                Conv2d(channel_list[1], channel_list[2], self.kernel_size, self.stride, self.feature_list[2]),
                Conv2d(channel_list[2], channel_list[2], self.kernel_size, self.stride, self.feature_list[3]),
                Conv2d(channel_list[2], channel_list[3], self.kernel_size, self.stride, self.feature_list[4])]

    def UNetBlock_de(self, channel_list):
        
        return [ConvTranspose2d(channel_list[3], channel_list[2], self.kernel_size, self.stride, self.feature_list[3], 'cat'),
                ConvTranspose2d(channel_list[2], channel_list[2], self.kernel_size, self.stride, self.feature_list[2], 'cat'),
                ConvTranspose2d(channel_list[2], channel_list[1], self.kernel_size, self.stride, self.feature_list[1], 'cat'),
                ConvTranspose2d(channel_list[1], channel_list[1], self.kernel_size, self.stride, self.feature_list[0], 'cat')]
    
class U2Net(nn.Module):
    def __init__(self,
                 channel_list=(2, 16, 32, 64),
                 kernel_size1=(2, 5),
                 kernel_size2=(2, 3),
                 kernel_size3=(2, 3),
                 kernel_size4=(2, 3),
                 kernel_size5=(2, 3),
                 kernel_size_path=(1, 3),
                 stride_path=(1, 2),
                 feature_list=(79, 39, 19, 9, 4),
                 dilation_list=(1, 2, 5, 9)):
        super(U2Net, self).__init__()
        
        u2net_list = []
        u2net_list.append(U2NetModule(in_channels=channel_list[0], 
                                      out_channels=channel_list[1],  
                                      kernel_size=kernel_size1, 
                                      kernel_size_path=kernel_size_path,
                                      stride_path=stride_path,
                                      channel_list=channel_list,
                                      feature_list=feature_list,
                                      num_blocks=4,
                                      dilation=dilation_list[0]))
        u2net_list.append(U2NetModule(in_channels=channel_list[1], 
                                      out_channels=channel_list[1],  
                                      kernel_size=kernel_size2, 
                                      kernel_size_path=kernel_size_path,
                                      stride_path=stride_path,
                                      channel_list=channel_list,
                                      feature_list=feature_list,
                                      num_blocks=3,
                                      dilation=dilation_list[1]))
        u2net_list.append(U2NetModule(in_channels=channel_list[1], 
                                      out_channels=channel_list[2],  
                                      kernel_size=kernel_size3, 
                                      kernel_size_path=kernel_size_path,
                                      stride_path=stride_path,
                                      channel_list=channel_list,
                                      feature_list=feature_list,
                                      num_blocks=2,
                                      dilation=dilation_list[2]))
        u2net_list.append(U2NetModule(in_channels=channel_list[2], 
                                      out_channels=channel_list[2],  
                                      kernel_size=kernel_size4, 
                                      kernel_size_path=kernel_size_path,
                                      stride_path=stride_path,
                                      channel_list=channel_list,
                                      feature_list=feature_list,
                                      num_blocks=1,
                                      dilation=dilation_list[3]))
        u2net_list.append(GLUConv2d(in_channels=channel_list[2], 
                                    out_channels=channel_list[3], 
                                    kernel_size=kernel_size5, 
                                    stride=(1, 2),
                                    feature_size=feature_list[-1]))
        self.u2net = nn.Sequential(*u2net_list)
        
    def forward(self, x):
        
        return self.u2net(x)  

class FilterBlock(nn.Module):
    
    def __init__(self, 
                 channels=161,
                 num_features=256,
                 hidden_size=64,
                 dilations=(1, 2, 5, 9), 
                 num_repeats=2):
        super(FilterBlock, self).__init__()
        
        self.in_channels_1 = channels+num_features
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv1d(in_channels=self.in_channels_1, out_channels=num_features, kernel_size=1)
        self.stcm1 = STCM(channels=num_features, 
                          hidden_size=hidden_size, 
                          dilations=dilations, 
                          num_repeats=num_repeats)
        self.conv1o = nn.Conv1d(in_channels=num_features, out_channels=channels, kernel_size=1)
        
    def forward(self, feat, x_mag):
        
        # (B F T)
        mag = torch.cat([feat, x_mag], dim=1)
        mag = self.conv1(mag)
        g = self.sigmoid(self.conv1o(self.stcm1(mag)))
        
        return g

class RefineBlock(nn.Module):
    
    def __init__(self, 
                 channels=161,
                 num_features=256,
                 hidden_size=64,
                 dilations=(1, 2, 5, 9), 
                 num_repeats=2):
        super(RefineBlock, self).__init__()
        
        self.in_channels_2 = 2*channels+num_features
        self.sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv1d(in_channels=self.in_channels_2, out_channels=num_features, kernel_size=1)
        self.stcm2_r = STCM(channels=num_features, 
                            hidden_size=hidden_size, 
                            dilations=dilations, 
                            num_repeats=num_repeats)
        self.conv2_ro = nn.Conv1d(in_channels=num_features, out_channels=channels, kernel_size=1)
        self.stcm2_i = STCM(channels=num_features, 
                            hidden_size=hidden_size, 
                            dilations=dilations, 
                            num_repeats=num_repeats)
        self.conv2_io = nn.Conv1d(in_channels=num_features, out_channels=channels, kernel_size=1)
        
    def forward(self, feat, x_real, x_imag):
        
        ri = torch.cat([feat, x_real, x_imag], dim=1)
        ri = self.conv2(ri)
        
        f_real = self.conv2_ro(self.stcm2_r(ri))
        f_imag = self.conv2_io(self.stcm2_i(ri))
        
        return f_real, f_imag

class FRNetLite(nn.Module):
    def __init__(self,
                 win_len=320,
                 win_inc=160,
                 U2Net_channel_list1=(1, 16, 32, 64),
                 U2Net_channel_list2=(2, 16, 32, 64),
                 U2Net_kernel_size1=(2, 5),
                 U2Net_kernel_size2=(2, 3),
                 U2Net_kernel_size3=(2, 3),
                 U2Net_kernel_size4=(2, 3),
                 U2Net_kernel_size5=(2, 3),
                 U2Net_kernel_size_path=(1, 3),
                 U2Net_stride_path=(1, 2),
                 U2Net_feature_list=(79, 39, 19, 9, 4),
                 U2Net_dilation_list=(1, 2, 5, 9),
                 GGM_channels=161,
                 GGM_num_features=256,
                 GGM_hidden_size=64,
                 GGM_num_repeats=2):
        super(FRNetLite, self).__init__()
        
        self.win_len = win_len
        self.win_inc = win_inc
        self.mag_encoder = U2Net(U2Net_channel_list1,
                                 U2Net_kernel_size1,
                                 U2Net_kernel_size2,
                                 U2Net_kernel_size3,
                                 U2Net_kernel_size4,
                                 U2Net_kernel_size5,
                                 U2Net_kernel_size_path,
                                 U2Net_stride_path,
                                 U2Net_feature_list,
                                 U2Net_dilation_list)
        self.ri_encoder = U2Net(U2Net_channel_list2,
                                U2Net_kernel_size1,
                                U2Net_kernel_size2,
                                U2Net_kernel_size3,
                                U2Net_kernel_size4,
                                U2Net_kernel_size5,
                                U2Net_kernel_size_path,
                                U2Net_stride_path,
                                U2Net_feature_list,
                                U2Net_dilation_list)
        self.mag_decoder = FilterBlock(GGM_channels,
                                       GGM_num_features,
                                       GGM_hidden_size,
                                       U2Net_dilation_list, 
                                       GGM_num_repeats)
        self.ri_decoder = RefineBlock(GGM_channels,
                                      GGM_num_features,
                                      GGM_hidden_size,
                                      U2Net_dilation_list, 
                                      GGM_num_repeats)
        
    def forward(self, x):
        
        x = compress(stft(x, self.win_len, self.win_inc), 0.5)
        # (B 2*F T)
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        # (B 1 T F)
        x_mag = ((x_real**2+x_imag**2)**0.5).transpose(1, 2).contiguous().unsqueeze(1)
        # (B 2 T F)
        x_ri = torch.stack([x_real.transpose(1, 2).contiguous(), x_imag.transpose(1, 2).contiguous()], dim=1)
        
        # (B 64 T 4)
        mag_feat = self.mag_encoder(x_mag)
        mag_feat = mag_feat.transpose(2, 3).contiguous()
        # (B 256 T)
        mag_feat = mag_feat.reshape(mag_feat.size(0), -1, mag_feat.size(-1))
        
        # (B 64 T 4)
        ri_feat = self.ri_encoder(x_ri)
        ri_feat = ri_feat.transpose(2, 3).contiguous()
        # (B 256 T)
        ri_feat = ri_feat.reshape(ri_feat.size(0), -1, ri_feat.size(-1))
        
        x_mag = x_mag.squeeze(1).transpose(1, 2).contiguous()
        g = self.mag_decoder(mag_feat, x_mag)
        f_real, f_imag = self.ri_decoder(ri_feat, x_real, x_imag)
        s_real, s_imag = CRM(x_real, x_imag, g, f_real, f_imag)
        s = torch.cat([s_real, s_imag], dim=1)
        s = istft(compress(s, 2), self.win_len, self.win_inc)
        
        return s
    
def test_model(net):
    
    macs, params = get_model_complexity_info(net, (16000,), as_strings=False, print_per_layer_stat=False)
    x = torch.rand(1, 16000)
    T = 0
    times = 1
    for _ in range(times):
        T1 = time.perf_counter()
        y = net(x)
        T2 = time.perf_counter()
        T = T+T2-T1
    T = T/times
    print(f'macs: {macs/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')
    print('{} -> {}'.format(x.shape, y.shape))
    print(f'time: {T*100:.2f} ms')
    
if __name__ == '__main__':
    torch.manual_seed(777)
    net = FRNetLite()
    test_model(net)

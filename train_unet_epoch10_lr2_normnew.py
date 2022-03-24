#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:33:50 2020

@author: liql

input channel = 2
consider the real and imag part as 2 different channels

in this file, I don't read all data into memory at once time.
i read data in batches. 

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import struct
import time
import random
from unet_channel2 import CleanU_Net

running_start = time.time()

# Hyper Parameters
EPOCH = 300
BATCH_SIZE = 128
LR = 0.001         # learning rate
N_TEST_IMG = 3

nx = 256
nz = 256
nw = 128

device = torch.device('cuda')

trshot_start = 90
trshot_range = 400
devshot_start = 60
devshot_range = 30

print('trshot_start is %i' %trshot_start)
print('trshot_range is %i' %trshot_range)
print('devshot_start is %i' %devshot_start)
print('devshot_range is %i' %devshot_range)

fn_path = './trained_network/ratioricker/90-490_epoch10_lr2_normnew/adam/'
fn_func_tr = fn_path + 'fun_obj_tr_epoch_'+str(EPOCH)+'_batchsize_'+str(BATCH_SIZE)+'.txt'
fn_func_dev = fn_path + 'fun_obj_dev_epoch_'+str(EPOCH)+'_batchsize_'+str(BATCH_SIZE)+'.txt'
fn_savenet_last = fn_path + 'unet_epoch_'+str(EPOCH)+'_batchsize_'+str(BATCH_SIZE)+'.pkl'
fn_lossfig = fn_path + 'fun_obj_epoch_'+str(EPOCH)+'_batchsize_'+str(BATCH_SIZE)+'.png'


# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):
    
    def __init__(self, msdata_imag, msdata_real, cmdata_imag, cmdata_real, data_start, data_range):
        self.msdata_imag = msdata_imag
        self.msdata_real = msdata_real
        self.cmdata_imag = cmdata_imag
        self.cmdata_real = cmdata_real
        self.data_start = data_start
        self.data_size = data_range


        if not os.path.isfile(self.msdata_imag):
            print(self.msdata_imag + 'does not exist!')
        if not os.path.isfile(self.msdata_real):
            print(self.msdata_real + 'does not exist!')
        if not os.path.isfile(self.cmdata_imag):
            print(self.cmdata_imag + 'does not exist!')
        if not os.path.isfile(self.cmdata_real):
            print(self.cmdata_real + 'does not exist!')
        
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        idx2 = (self.data_start + np.arange(self.data_size))[idx]
     
        izslice_imag_ms = np.fromfile(self.msdata_imag, dtype=np.float32, count=nx*nw, offset=idx2*nx*nw*4).reshape(nx,nw)
        izslice_real_ms = np.fromfile(self.msdata_real, dtype=np.float32, count=nx*nw, offset=idx2*nx*nw*4).reshape(nx,nw)
        izslice_imag_cm = np.fromfile(self.cmdata_imag, dtype=np.float32, count=nx*nw, offset=idx2*nx*nw*4).reshape(nx,nw)
        izslice_real_cm = np.fromfile(self.cmdata_real, dtype=np.float32, count=nx*nw, offset=idx2*nx*nw*4).reshape(nx,nw)
        
        # convert the list to torch.Tensor. it's used for loading data to network.
        #record_msdata=torch.Tensor(np.array([izslice_real_ms,izslice_imag_ms]))   
        #record_cmdata=torch.Tensor(np.array([izslice_real_cm,izslice_imag_cm])) 
        maxamp = np.max([np.max(abs(izslice_real_ms)), np.max(abs(izslice_imag_ms))])  
        record_msdata=torch.Tensor(np.array([izslice_real_ms/maxamp, izslice_imag_ms/maxamp]))   
        record_cmdata=torch.Tensor(np.array([izslice_real_cm/maxamp, izslice_imag_cm/maxamp]))   

        return record_msdata, record_cmdata


# train data is only for six different velocity models    
train_dataset = MyDataset(
    '/data/ess-liql/ML_seismod/dataset_layermodel/missing_data_121_60_regular/allshotfile_imag.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/missing_data_121_60_regular/allshotfile_real.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/complete_data/allshotfile_imag.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/complete_data/allshotfile_real.bin',
    data_start = trshot_start*nz, data_range = trshot_range*nz
    )

# 使用DataLoader可以利用多线程，batch,shuffle等
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=16,  # whether more programmer
                          pin_memory=True)   

dev_dataset = MyDataset(
    '/data/ess-liql/ML_seismod/dataset_layermodel/missing_data_121_60_regular/allshotfile_imag.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/missing_data_121_60_regular/allshotfile_real.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/complete_data/allshotfile_imag.bin',
    '/data/ess-liql/ML_seismod/dataset_layermodel/complete_data/allshotfile_real.bin',
    data_start = devshot_start*nz, data_range = devshot_range*nz
    )

# 使用DataLoader可以利用多线程，batch,shuffle等
dev_loader = DataLoader(dataset=dev_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=16,  # whether more programmer
                          pin_memory=True)     



unet = CleanU_Net()
unet = unet.to(device)
print(unet)

running_end1 = time.time()
print("the time of readingfile is :%.2fs" %(running_end1-running_start))

###############################################################################
# training part
# model = torch.load('trained_network/120shots_10v_6m/adam_weight0/ixiz/unet_epoch_200_batchsize_128.pkl')
# unet = model.module.to(device)
# print(unet)

unet = nn.DataParallel(unet)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR, amsgrad=True)
# optimizer = torch.optim.SGD(unet.parameters(), lr=LR, momentum=0.9)
# optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-2)
print(optimizer)
loss_func = nn.MSELoss()


fun_obj_tr = []
fun_obj_dev = []

for epoch in range(EPOCH):
    for step, (record_ms_tr, record_cm_tr) in enumerate(train_loader):
        # print(record_ms_tr.size())
        # b_x = torch.unsqueeze(record_ms_tr,dim=1)   # change the dimension of torch.tensor from (batch_size,150,500) to (batch_size,1,150,500)
        # b_y = torch.unsqueeze(record_cm_tr,dim=1)
        b_x = record_ms_tr.to(device)
        b_y = record_cm_tr.to(device)
        
        output_unet= unet(b_x)
        loss_tr = loss_func(output_unet, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss_tr.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        
        
    print('Epoch: ', epoch, '| train loss: %0.4E' % loss_tr.data.cpu().numpy())
    fun_obj_tr.append(loss_tr.data.cpu().numpy())
    
    #if epoch%10==0 and epoch !=0:
    if 0:
        f1, a1 = plt.subplots(4, N_TEST_IMG, figsize=(4, N_TEST_IMG))
        for i in range(N_TEST_IMG):
            plt.rcParams['figure.dpi'] = 600 #分辨率
            plt.rcParams['font.size'] = 3  # fontsize
          
            dat_ms = np.transpose(b_x.cpu().numpy()[i,0,:,:])
            output_net = np.transpose(output_unet.detach().cpu().numpy()[i,0,:,:])
            dat_cm = np.transpose(b_y.cpu().numpy()[i,0,:,:])
          
            ## 2D imageshow
            a1[0][i].clear()
            im=a1[0][i].imshow(dat_ms, aspect='auto')
            f1.colorbar(im, ax=a1[0][i])
              
            a1[1][i].clear()
            im=a1[1][i].imshow(dat_cm, aspect='auto')
            f1.colorbar(im, ax=a1[1][i])
          
            a1[2][i].clear()
            im=a1[2][i].imshow(output_net, aspect='auto')
            f1.colorbar(im, ax=a1[2][i])
          
            a1[3][i].clear()
            im=a1[3][i].imshow(output_net-dat_cm, aspect='auto')
            f1.colorbar(im, ax=a1[3][i])
            snr = 2*10*np.log10(np.linalg.norm(dat_cm)/np.linalg.norm(output_net-dat_cm))
            snr=np.around(snr,decimals=2)
            a1[3][i].annotate('the SNR = ' + str(snr), (30,200) )
          
            if i != 0:
                a1[0][i].set_xticks(()); a1[0][i].set_yticks(());
                a1[1][i].set_xticks(()); a1[1][i].set_yticks(());
                a1[2][i].set_xticks(()); a1[2][i].set_yticks(());
                a1[3][i].set_yticks(());
    
        f2, a2 = plt.subplots(4, N_TEST_IMG, figsize=(4, N_TEST_IMG))
        for i in range(N_TEST_IMG):
            plt.rcParams['figure.dpi'] = 600 #分辨率
            plt.rcParams['font.size'] = 3  # fontsize
          
            dat_ms = np.transpose(b_x.cpu().numpy()[i,1,:,:])
            output_net = np.transpose(output_unet.detach().cpu().numpy()[i,1,:,:])
            dat_cm = np.transpose(b_y.cpu().numpy()[i,1,:,:])
          
            ## 2D imageshow
            a2[0][i].clear()
            im=a2[0][i].imshow(dat_ms, aspect='auto')
            f2.colorbar(im, ax=a2[0][i])
              
            a2[1][i].clear()
            im=a2[1][i].imshow(dat_cm, aspect='auto')
            f2.colorbar(im, ax=a2[1][i])
          
            a2[2][i].clear()
            im=a2[2][i].imshow(output_net, aspect='auto')
            f2.colorbar(im, ax=a2[2][i])
          
            a2[3][i].clear()
            im=a2[3][i].imshow(output_net-dat_cm, aspect='auto')
            f2.colorbar(im, ax=a2[3][i])
            snr = 2*10*np.log10(np.linalg.norm(dat_cm)/np.linalg.norm(output_net-dat_cm))
            snr=np.around(snr,decimals=2)
            a2[3][i].annotate('the SNR = ' + str(snr), (30,200) )
          
            if i != 0:
                a2[0][i].set_xticks(()); a2[0][i].set_yticks(());
                a2[1][i].set_xticks(()); a2[1][i].set_yticks(());
                a2[2][i].set_xticks(()); a2[2][i].set_yticks(());
                a2[3][i].set_yticks(());

    for step, (record_ms_dev, record_cm_dev) in enumerate(dev_loader):
        # print(record_ms_tr.size())
        #b_x = torch.unsqueeze(record_ms_dev,dim=1)   # change the dimension of torch.tensor from (batch_size,150,500) to (batch_size,1,150,500)
        #b_y = torch.unsqueeze(record_cm_dev,dim=1)
        b_x = record_ms_dev.to(device)
        b_y = record_cm_dev.to(device)
        
        output_unet= unet(b_x)
        loss_dev = loss_func(output_unet, b_y)      # mean square error
    print('Epoch: ', epoch, '| develop loss: %0.4E' % loss_dev.data.cpu().numpy())
    fun_obj_dev.append(loss_dev.data.cpu().numpy())
    
    #if epoch%10 == 0 or epoch == EPOCH:
    if 0:
        plt.figure()
        plt.rcParams['figure.dpi'] = 600 #分辨率
        plt.rcParams['font.size'] = 12  # fontsize
        line1, = plt.plot(fun_obj_tr)
        line2, = plt.plot(fun_obj_dev)
        plt.xlabel('Epoch');plt.ylabel('Loss function')
        plt.legend([line1,line2], ('loss_train','loss_dev'))
        plt.savefig(fn_lossfig,dpi=600)
    
    
    if epoch%20 == 0 or epoch == max(range(EPOCH)):
        fn_savenet = fn_path + 'unet_epoch_'+str(epoch)+'_batchsize_'+str(BATCH_SIZE)+'.pkl'
        torch.save(unet, fn_savenet)  # save network 
        
        np.savetxt(fn_func_tr, fun_obj_tr, fmt="%0.4E", delimiter=',')
        np.savetxt(fn_func_dev, fun_obj_dev, fmt="%0.4E", delimiter=',')

    if epoch%10==0 and epoch !=0:
        optimizer = torch.optim.Adam(unet.parameters(), lr=LR/int(epoch/10)/2, amsgrad=True)
        print(optimizer)

torch.save(unet, fn_savenet_last)  # save last network



running_end2 = time.time()
print("the epoch running period is :%.2fs" %(running_end2-running_end1))
print("the whole running period is :%.2fs" %(running_end2-running_start))







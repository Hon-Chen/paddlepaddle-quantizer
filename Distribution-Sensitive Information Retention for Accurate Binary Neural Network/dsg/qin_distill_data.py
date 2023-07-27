#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*
import cv2
cv2.setNumThreads(0)
import os
import json
import math
import numpy
from PIL import Image
import torch
import random
import torch.nn as nn
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
from tensorboardX import SummaryWriter
from torchvision import transforms

def own_loss(A, B, s, cnt=-1, onehot=0, layer_num=20):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
    """
    m = s*torch.ones(A.size()).cuda()
    C = (A - B).abs() - m.abs()
    zero = torch.zeros(C.size()).cuda()
    D = torch.max(C, zero)
    if cnt != -1 and onehot != 0:
        for i in range(max(int(D.size(0)/2), layer_num)):
            if i % layer_num == cnt:
                D[i] = D[i] * math.sqrt(2)
    if cnt == 0:
        return D.norm()**2 / B.size(0)
    else:
        return D.norm()**2 / B.size(0)
    #return (A - B).norm()**2 / B.size(0)


class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer.
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
                         diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def getDistilData(args,
                  teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  for_inception=False):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""

    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)

    #if args.one_hot:
    #    num_batch = 20
    #else:
    #    num_batch = 1
    #print('num_batch:', num_batch)
    #print('batch_size:', batch_size)
    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    #dataloader = getTestData(dataset,
    #                          batch_size=batch_size,
    #                          path='/home/dell/imagenet/val',
    #                          for_inception=for_inception)
    '''
    for i, (data, target) in enumerate(true_loader):
        #writer = SummaryWriter('./truelog')
        if i == num_batch:
            break
        data = data.cuda()
        data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()
        for hook in hooks:
            hook.clear()
        output = teacher_model(data)
        mean_loss = 0
        std_loss = 0
        mean_l = []
        std_l = []
        mean = []
        std = []
        for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
            tmp_output = hook.outputs
            bn_mean, bn_std = bn_stat[0], bn_stat[1]
            print(bn_mean.shape)
            mean.append(bn_mean.mean().cpu().item())
            std.append(bn_std.mean().cpu().item())
            tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),tmp_output.size(1), -1),dim=2)
            tmp_std = torch.sqrt(torch.var(tmp_output.view(tmp_output.size(0),tmp_output.size(1), -1),dim=2) + eps)
            mean_l.append(own_loss(bn_mean, tmp_mean).cpu().item())
            std_l.append(own_loss(bn_std, tmp_std).cpu().item())
            mean_loss += own_loss(bn_mean, tmp_mean)
            std_loss += own_loss(bn_std, tmp_std)
        tmp_mean = torch.mean(data.view(data.size(0), 3, -1),dim=2)
        tmp_std = torch.sqrt(torch.var(data.view(data.size(0), 3, -1),dim=2) + eps)
        mean_loss += own_loss(input_mean, tmp_mean)
        std_loss += own_loss(input_std, tmp_std)
        total_loss = mean_loss + std_loss
        print(mean_loss,std_loss,total_loss)
        fig = plt.figure(figsize=(30,15),dpi=80)
        x = range(0, len(mean_l))
        plt.xticks(x[::1])
        #plt.scatter(x, mean_l, color = 'red', s = 100, label = 'mean_loss')
        #plt.scatter(x, std_l, color = 'blue', s = 100, label = 'std_loss')
        plt.scatter(x, mean, color = 'black', s = 100, label = 'bn_mean')
        plt.scatter(x, std, color = 'green', s = 100, label = 'bn_std')
        plt.legend()
        plt.savefig('sqbn_stats.jpg')
        plt.show()
        #niter = it
        #writer.add_scalars('./log/' + 'Train_val_loss', {'./log/'+'mean_loss': mean_loss.data.item()}, niter)
        #writer.add_scalars('./log/' + 'Train_val_loss', {'./log/'+'std_loss': std_loss.data.item()}, niter)
    '''
    label = []
    features = {}
    for i, gaussian_data in enumerate(dataloader):
        #writer = SummaryWriter('./log')
        if i == num_batch:
            break
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        labels = Variable(torch.randint(low=0, high=1000, size=(batch_size,)))
        label.append(labels)
        labels = labels.cuda()
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.1)
        it_num = 500
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-6,
                                                         verbose=True,
                                                         patience=100)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        #                                                 it_num, 
        #                                                 eta_min = 0, 
        #                                                 last_epoch=-1)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()
        lim_0 = 30
        for it in range(it_num):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0
            mean_l = []
            std_l = []
            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            if args.m_n_perc != 0:
                #m = 5
                #n=1.5
                m = np.load('./m_n/m_' + str(args.m_n_perc) + '.npy', allow_pickle = True)  
                n = np.load('./m_n/n_' + str(args.m_n_perc) + '.npy', allow_pickle = True) 
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                #if it == 499:
                #    if i == 0:
                #        features[cnt] = []
                #    features[cnt].append(hook.outputs.detach().clone().cpu().numpy())
                #    if i == 0:
                #        output_1 = np.concatenate(features[cnt])
                #        print(output_1.shape)
                #        np.save('./t_features/' + str(cnt) + '.npy', output_1)
                # if i < num_batch / 2:
                #     if i == cnt:
                #         s = 1 + args.onehot
                #     else:
                #         s = 1
                # else:
                #     s = 1
                s = 1
                if args.m_n_perc == 0:
                    mean_loss += s * own_loss(bn_mean, tmp_mean, 0, cnt=cnt, onehot=args.onehot)
                    std_loss += s * own_loss(bn_std, tmp_std, 0, cnt=cnt, onehot=args.onehot)
                else:
                    #mean_loss += s * own_loss(bn_mean, tmp_mean, 5, cnt=cnt, onehot=args.onehot)
                    #std_loss += s * own_loss(bn_std, tmp_std, 1.5, cnt=cnt, onehot=args.onehot)
                    
                    mean_loss += s * own_loss(bn_mean, tmp_mean, m[cnt], cnt=cnt, onehot=args.onehot)
                    std_loss += s * own_loss(bn_std, tmp_std, n[cnt], cnt=cnt, onehot=args.onehot)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,
                                                     -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1),
                          dim=2) + eps)
            mean_loss += own_loss(input_mean, tmp_mean, 0)
            std_loss += own_loss(input_std, tmp_std, 0)
            off1 = torch.randint(-lim_0, lim_0, size = (1,)).item()
            inputs_jit = torch.roll(gaussian_data, shifts=(off1, off1), dims=(2, 3)) 
            off1 = torch.randint(-lim_0, lim_0, size = (1,))
            loss_l2 = torch.norm(inputs_jit.view(batch_size, -1), dim=1).mean()
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit) # 只用了 loss_var_l2
            print("dyf: image prior")

            loss_labels = crit(output, labels)

            #total_loss = args.a*std_loss
            total_loss = mean_loss + args.a*std_loss
            if args.label:
                total_loss += 10*loss_labels
            if args.deepinversion:
                total_loss += (10*loss_labels + 0.001*loss_var_l2 + 0.001*loss_l2)


            if (it+1) % 100 == 0:
                print('iter:', it)
                print('mean_loss:', mean_loss.item())
                print('std_loss:', std_loss.item())
                #print('loss_labels:', loss_labels.item())
                #print('loss_l2:', loss_l2.item())
                #print('loss_var_l1:', loss_var_l1.item())
                #print('loss_var_l2:', loss_var_l2.item())

            
            #if it == 499:
            #    print(mean_loss, std_loss)
                #fig = plt.figure(figsize=(20,10),dpi=50)
                #x = range(0, len(mean_l))
                #plt.xticks(x[::1])
                #plt.scatter(x, mean_l, color = 'red', s = 100, label = 'mean_loss')
                #plt.scatter(x, std_l, color = 'blue', s = 100, label = 'std_loss')
                #plt.legend()
                #plt.savefig('./full_loss/' + str(i) + 'layer_loss.jpg')
                #plt.show()
            #niter = it
            #writer.add_scalars(str(a)+'2Train_val_loss', {'./'+str(a)+'/'+'mean_loss': mean_loss.data.item()}, niter)
            #writer.add_scalars(str(a)+'2Train_val_loss', {'./'+str(a)+'/'+'std_loss': std_loss.data.item()}, niter)
            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    #print(refined_gaussian[0].shape)
    #import matplotlib
    ##unloader = transforms.ToPILImage()
    #for j in range(0, num_batch):
    #    for i in range(0, batch_size):
    #        image = refined_gaussian[j][i].cpu().clone().numpy()
    #        image = image.transpose(1, 2, 0)
    #        image = (image - image.min(axis = (0,1)))/(image.max(axis = (0,1)) - image.min(axis = (0,1)))
    #        #print(image.shape)
    #        matplotlib.image.imsave('./zeroqimg/' + str(j) + '_' + str(i) + '.pdf', image)
    return refined_gaussian

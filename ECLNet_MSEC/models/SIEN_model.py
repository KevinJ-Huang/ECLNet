import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,AMPLoss,PhaLoss,FourierLoss
from models.loss_new import SSIMLoss,VGGLoss
import torch.nn.functional as F
import random
from metrics.calculate_PSNR_SSIM import psnr_np
import cv2
logger = logging.getLogger('base')
from scipy import signal
import numpy as np

class GaussianBlur(nn.Module):

    def __init__(self):
        super(GaussianBlur, self).__init__()

        def get_kernel(size=51, std=3):
            """Returns a 2D Gaussian kernel array."""
            k = signal.gaussian(size, std=std).reshape(size, 1)
            k = np.outer(k, k)
            return k/k.sum()

        kernel = get_kernel(size=51, std=3)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0).repeat([3, 1, 1, 1])
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)  # shape [3, 1, 11, 11]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
        """
        x = F.conv2d(x, self.kernel, padding=25, groups=3)
        return x



class SIEN_Model(BaseModel):
    def __init__(self, opt):
        super(SIEN_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']
        self.blur = GaussianBlur().cuda()
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()


        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
                # self.cri_amp = AMPLoss().to(self.device)
                # self.cri_phase = PhaLoss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        LQ_IMG = data['LQ']
        GT_IMG = data['GT']
        # LQright_IMG = data['LQright']
        self.var_L = LQ_IMG.to(self.device)
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0


    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.netG.zero_grad() ################################################# new add
        self.optimizer_G.zero_grad()


        # LR_right = self.varright_L
        out,feature = self.netG(self.var_L)
        gt = self.real_H

        if self.opt['train']['augment']:
            maskzeros = torch.zeros_like(self.var_L)
            randxstart = random.randrange(10, 270, 1)
            randystart = random.randrange(10, 270, 1)
            maskzeros[:, :, randxstart:randxstart + 100, randystart:randystart + 100] = 1.0

            alpha = random.randrange(10, 50, 2) / 100.0
            augori = (1-alpha) * self.var_L + alpha * self.real_H
            maskblur = self.blur(maskzeros)
            maskblur = maskblur.to(self.device)
            auginput = maskblur * augori + (1-maskblur) * self.var_L

            _, augfeature = self.netG(auginput)
            l_aux = self.cri_pix(feature,augfeature)


        l_total = self.cri_pix(out, gt)+0.1*self.cri_ssim(out, gt)

        if self.opt['train']['augment']:
            l_total = l_total+0.02*l_aux

        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = out
        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())

        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['l_total'] = l_total.item()
        # self.log_dict['output_is_nan'] = torch.isnan(torch.mean(out))


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            out, x_pre,x1 = self.netG(self.var_L)
            self.fake_H = out

        self.netG.train()



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])


    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self,name):
        self.save_network(self.netG, 'best'+name, 0)

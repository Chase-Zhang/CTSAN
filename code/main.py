#!/usr/bin/env python
# coding=gbk


##############################################################
# Our code was originally developed from CDVD-TSP and EDVR:
#       (1) Jinshan Pan, Haoran Bai, Jinhui Tang. "Cascaded Deep Video Deblurring Using Temporal Sharpness Prior", CVPR2020.
#       (2) Xintao Wang, Kelvin C.K. Chan, Ke Yu, Chao Dong, Chen Change Loy. "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks", CVPRW2019.
# We sincerely thank their contributions in publicy available codes.
##############################################################


import torch
import data
import model
import loss
import option
from trainer.trainer_ctsan import Trainer_CTSAN
from logger import logger

args = option.args
torch.manual_seed(args.seed) 
chkp = logger.Logger(args)




if args.task == 'VideoDeblur':
    print("Selected task: {}".format(args.task))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = Trainer_CTSAN(args, loader, model, loss, chkp) 
    while not t.terminate():
        t.train()
        t.test()
        


chkp.done()



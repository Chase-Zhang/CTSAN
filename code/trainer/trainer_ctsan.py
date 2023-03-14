import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import os
import numpy as np



class Trainer_CTSAN(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_CTSAN, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CTSAN")
        assert args.n_sequence == 5, \
            "Only support args.n_sequence=5; but get args.n_sequence={}".format(args.n_sequence)

    def make_optimizer(self): #
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam([{"params": self.model.get_model().recons_net.parameters()},
                           {"params": self.model.get_model().tsa_fusion.parameters()},
                           {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
                                **kwargs)

    def train(self):

        print("Now training")
        self.loss.step() # step():
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        
       
        for param in self.model.parameters():
            mulValue = np.prod(param.size())  
            Total_params += mulValue 
            if param.requires_grad:
                Trainable_params += mulValue  
            else:
                NonTrainable_params += mulValue  
        
        # print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')
	

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            gt_list = [gt[:, i, :, :, :] for i in range(self.args.n_sequence)]
            gt = torch.cat([gt_list[1], gt_list[2], gt_list[3], gt_list[2]], dim=1).to(self.device) 

            recons_1, recons_2, recons_3, recons_2_iter, mid_loss = self.model(input)
            output = torch.cat([recons_1, recons_2, recons_3, recons_2_iter], dim=1)

            self.optimizer.zero_grad() # Clears the gradients of all optimized.
            loss = self.loss(output, gt) 
            if mid_loss:  
                loss = loss + self.args.mid_loss_weight * mid_loss
                mid_loss_sum = mid_loss_sum + mid_loss.item()
            loss.backward()
            self.optimizer.step()
            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1)
                ))
        self.scheduler.step()
        self.loss.end_log(len(self.loader_train))
        
        
        


    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_PSNR_iter1 = 0.
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                recons_1, recons_2, recons_3, recons_2_iter, _ = self.model(input)

                PSNR_iter1 = utils.calc_psnr(gt, recons_2, rgb_range=self.args.rgb_range)
                total_PSNR_iter1 += PSNR_iter1
                total_num += 1
                PSNR = utils.calc_psnr(gt, recons_2_iter, rgb_range=self.args.rgb_range)
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:

                    gt, input_center, recons_2, recons_2_iter = utils.postprocess(gt, input_center, recons_2,
                                                                                  recons_2_iter,
                                                                                  rgb_range=self.args.rgb_range,
                                                                                  ycbcr_flag=False, device=self.device)
                    save_list = [gt, input_center, recons_2, recons_2_iter]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR_iter1: {:.3f} PSNR_iter2: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                total_PSNR_iter1 / total_num,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch=epoch, is_best=(best[1] + 1 == epoch))


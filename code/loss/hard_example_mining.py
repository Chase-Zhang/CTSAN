import torch
import torch.nn as nn
import numpy as np


class HEM(nn.Module): # HEM: hard example mining.
    def __init__(self, hard_thre_p=0.5, device='cuda', random_thre_p=0.1): # hard mask, top 0.5; random mask, random 0.1.
        super(HEM, self).__init__()
        self.hard_thre_p = hard_thre_p
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()
        self.device = device

    def hard_mining_mask(self, x, y): # x and y represent blurred images and ground truth.
        with torch.no_grad():
            b, c, h, w = x.size() # number,channel,length and width.

            # hard mask.
            hard_mask = np.zeros(shape=(b, 1, h, w))
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
            res_numpy = res.cpu().numpy()
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)]
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind].item()
                hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

            # random mask.
            random_thre_ind = int(self.random_thre_p * w * h)
            random_mask = np.zeros(shape=(b, 1 * h * w))
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                np.random.shuffle(random_mask[i])
            random_mask = np.reshape(random_mask, (b, 1, h, w))


            mask = hard_mask + random_mask
            mask = (mask > 0.).astype(np.float32)
            mask = torch.Tensor(mask).to(self.device)

        return mask

    def forward(self, x, y): 
        mask = self.hard_mining_mask(x.detach(), y.detach()).detach() 

        hem_loss = self.L1_loss(x * mask, y * mask)

        return hem_loss 

    

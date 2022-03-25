# key point variance loss
import os 
import torch
from torch.functional import norm
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import WarpingGAN, Discriminator

from gradient_penalty import GradientPenalty
from data_benchmark import BenchmarkDataset
from stitchingloss import stitchloss
from shutil import copyfile

from arguments import Arguments

import time
import visdom
import numpy as np

class WarpingGANTrain():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, class_choice=args.class_choice)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = WarpingGAN(num_points=2048).to(args.device)      
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             

        # -------------------------------------------------adam---------------------------------------------- #
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        self.vis = visdom.Visdom(port=args.visdom_port)
        assert self.vis.check_connection()
        print("Visdom connected.")
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):        
        color_num = self.args.visdom_color
        chunk_size = int( 2048 / color_num)
        colors = np.array([(227,0,27),(231,64,28),(237,120,15),(246,176,44),
                           (252,234,0),(224,221,128),(142,188,40),(18,126,68),
                           (63,174,0),(113,169,156),(164,194,184),(51,186,216),
                           (0,152,206),(16,68,151),(57,64,139),(96,72,132),
                           (172,113,161),(202,174,199),(145,35,132),(201,47,133),
                           (229,0,123),(225,106,112),(163,38,42),(128,128,128)])
        colors = colors[np.random.choice(len(colors), color_num, replace=False)]
        label = torch.stack([torch.ones(chunk_size).type(torch.LongTensor) * inx for inx in range(1,int(color_num)+1)], dim=0).view(-1)
        # label = label[:-3]

        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        

        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            
            print("Checkpoint loaded.")

        for epoch in range(epoch_log, self.args.epochs):
            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                
                point, _ = data
                point = point.to(self.args.device)
                start_time = time.time()
                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()

                    z = torch.randn(self.args.batch_size, 1, 128).to(self.args.device)

                    with torch.no_grad():
                        fake_point = self.G(z)         
                        fake_point = (fake_point)

                    D_real, real_index = self.D(point)
                    D_realm = D_real.mean()
                    D_fake, _ = self.D(fake_point)
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                realvar = stitchloss(point, real_index)

                loss_log['D_loss'].append(d_loss.item())                  
                
                # ---------------------- Generator ---------------------- #
                
                self.G.zero_grad()
                z = torch.randn(self.args.batch_size, 1, 128).to(self.args.device)
                fake_point = self.G(z)
                fake_point = (fake_point)
                G_fake, fake_index = self.D(fake_point)
                
                fakevar = stitchloss(fake_point,fake_index)
                G_fakem = G_fake.mean()
                
                varloss = torch.pow((fakevar-realvar),2)
                
                g_loss = -G_fakem + 0.05*varloss
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())
                 
                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))

                if _iter % 4 == 0:
                    generated_point = fake_point[-1]
                    # print(generated_point.shape)
                    # print(label.shape)
                    plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)

                    self.vis.line(X=plot_X, Y=plot_Y, win=1,
                                  opts={'title': 'WarpingGAN Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})

                    self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], Y=label, win=2,
                                     opts={'title': "Generated Pointcloud", 'markersize': 3, 'markercolor': colors, 'webgl': True})
                     

                    print('Figures are saved.')
            
            if epoch % 100 == 0 and not result_path == None:
                fake_pointclouds = torch.Tensor([])
                for i in range(250): # For 5000 samples
                    z = torch.randn(self.args.batch_size, 1, 128).to(self.args.device)
                    with torch.no_grad():
                        sample = self.G(z).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                class_name = args.class_choice if args.class_choice is not None else 'all'
                torch.save(fake_pointclouds, result_path+str(epoch)+'_'+class_name+'.pt')
                del fake_pointclouds

            # ---------------------- Save checkpoint --------------------- #
            if epoch % 50 == 0 and not save_ckpt == None:
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                }, save_ckpt+str(epoch)+'.pt')

                print('Checkpoint is saved.')
            
                
                    

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None
    RESULT_PATH = args.result_path + args.result_save
     

    model = WarpingGANTrain(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)

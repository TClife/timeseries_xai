import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torch.utils.data import Dataset, random_split
from dalle_vqvae import VQVAE
import numpy as np
from scipy.io.arff import loadarff
import argparse
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt 
import os
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()
wandb.init()
torch.manual_seed(911)

def str2bool(v):
    return v.lower() in ('true')
 
#Trainer 
class VQTrainer():
    def __init__(self, args):
        self.args = args

        ecg_train = np.loadtxt("./data/UCR_official/CinCECGTorso_TRAIN.txt")
        ecg_test =np.loadtxt("./data/UCR_official/CinCECGTorso_TEST.txt")

        ecg_train = torch.tensor(ecg_train)
        ecg_test = torch.tensor(ecg_test)

        ecg_train_y = ecg_train[:, 0]
        ecg_train_x = ecg_train[:, 1:1633]
        ecg_test_y = ecg_test[:, 0]
        ecg_test_x = ecg_test[:, 1:1633]

        #Dataset
        data = torch.cat((ecg_train_x, ecg_test_x), dim=0)


        labels = torch.cat((ecg_train_y, ecg_test_y), dim=0)
        labels = labels - 1


        class ECGDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
            
        ds = ECGDataset(data, labels)

        # Split the dataset into training, validation, and test sets
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    def train(self, vae):
        self.model = vae.to(device)
        optimizer = optim.Adam(self.model.parameters(), self.args.lr, amsgrad=False)
        scheduler = MultiStepLR(optimizer, milestones=[150], gamma=0.2)
        train_res_recon_error = []
        val_res_recon_error = []
        train_vq_loss_list = []
        val_vq_loss_list = []

        best_val_mse = 1000
        best_model = None

        #directory 
        self.savedir = self.args.savedir
        directory = os.path.join(self.savedir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for epoch in range(self.args.n_epochs):
            self.model.train()
            train_epoch_mse, train_vq_loss = 0.0, 0.0

            for i,(data,_) in enumerate(self.train_loader):
                data = data.to(device).unsqueeze(1)
                data = data.float()

                optimizer.zero_grad()

                data_recon, recon_error, vq_loss= self.model(data)

                loss = recon_error + vq_loss
                #backprop
                loss.backward()
                optimizer.step()

                train_epoch_mse += recon_error.item()
                train_vq_loss += vq_loss.item()

            train_res_recon_error.append(train_epoch_mse/len(self.train_loader))
            train_vq_loss_list.append(train_vq_loss/len(self.train_loader))
            wandb.log({"Training recon loss": train_epoch_mse/len(self.train_loader), "Training vq loss": train_vq_loss/len(self.train_loader)})

            scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_epoch_mse, val_vq_loss = 0.0, 0.0

                for j,(data,_) in enumerate(self.val_loader):
                    data = data.to(device).unsqueeze(1)
                    data = data.float()
                    data_recon, recon_error, vq_loss = self.model(data)
                    loss = recon_error + vq_loss

                    val_epoch_mse += recon_error.item()
                    val_vq_loss += vq_loss.item()

                    if j==0:
                        plt.plot(data[j, 0, :].cpu().numpy(), label='original')
                        plt.plot(data_recon[j, 0, :].cpu().numpy(), label='recon')
                        plt.legend()
                        plt.savefig(f"{self.savedir}/recon.png")
                        plt.clf()

                val_res_recon_error.append(val_epoch_mse/len(self.val_loader))
                val_vq_loss_list.append(val_vq_loss/len(self.val_loader))
                wandb.log({"Validation recon loss": val_epoch_mse/len(self.val_loader), "Validation vq loss": val_vq_loss/len(self.val_loader)})
                

            print(f'epoch[{epoch}]', f'train_loss: {train_res_recon_error[-1]:.6f}',
                  f'\tval_loss: {val_res_recon_error[-1]:.6f}')

            if best_val_mse > val_res_recon_error[-1]:
                best_val_mse = val_res_recon_error[-1]
                best_model = copy.deepcopy(self.model)

                savedict = {
                'args': self.args,
                'model_state_dict': best_model.state_dict(),
                'bpe_vocab': None
                    }
                
            if epoch % 10 == 0:
                torch.save(savedict, f'{self.savedir}/model_{epoch}.pt')                

        best_iter = np.argmin(val_res_recon_error)
        print('try to save in testdir')
        print(best_iter, f'save ... {self.savedir}/')
        print(f'train: {train_res_recon_error[best_iter]:.4f}\t')
        print(f'val: {val_res_recon_error[best_iter]:.4f}\t')
        print("save done!")
    
    def test(self):
        #load model
        a = torch.load(args.load_model)

        self.model.load_state_dict(a['model_state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False 

        self.model.eval()

        perturb_codebooks = torch.load(args.test_data)
        perturb_codebooks = torch.cat(perturb_codebooks)
        data = perturb_codebooks.to(device)

        reconstruct = self.model.code_decode(data)

        recon_error = F.mse_loss(reconstruct, data)

        return reconstruct #save decoded codebook

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    #Model Configuration
    parser.add_argument('--ecg_size', type=int, default=1632, help="Number of timesteps of ECG")
    parser.add_argument('--num_layers', type=int, default=5, help="Number of convolutional layers")
    parser.add_argument('--num_tokens', type=int, default=64, help="Number of tokens in VQVAE")
    parser.add_argument('--codebook_dim', type=int, default=64, help="Dimension of codebook")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Dimension of hidden layer")
    parser.add_argument('--num_resnet_blocks', type=int, default=0, help="Number of resnet blocks")
    parser.add_argument('--temperature', type=float, default=0.9, help="Temperature for gumbel softmax")
    parser.add_argument('--straight_through', type=bool, default=False, help="Straight through estimator for gumbel softmax")

    parser.add_argument('--savedir', type=str, default="./vqvae_model")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])

    #if test
    parser.add_argument('--test_data', type=str, default=False, help="Quantized test data for decoding")
    parser.add_argument('--load_model', type=str, default=False, help="VQVAE model path")

    args = parser.parse_args()

    vqtrain = VQTrainer(args)

    #load vqvae model 
    vae = VQVAE(
    image_size = args.ecg_size,
    num_layers = args.num_layers,                 # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = args.num_tokens,                 # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = args.codebook_dim,             # codebook dimension
    hidden_dim = args.hidden_dim,                 # hidden dimension
    num_resnet_blocks = args.num_resnet_blocks,   # number of resnet blocks
    temperature = args.temperature,               # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = args.straight_through      # straight-through for gumbel softmax. unclear if it is better one way or the other
    ).to(device)

    if args.mode == "train":
        vqtrain.train(vae)
    
    if args.mode =="test":
        vqtrain.test()



    


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torch.utils.data import Dataset, random_split
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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, kernel_size, stride_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride_size, padding=0)
        self.conv6 = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=kernel_size, stride=stride_size, padding=0)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.float()
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_nor):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.use_nor = use_nor

    def infer(self, encoding_indices):
        with torch.no_grad():
            if self.use_nor:
                i_mean = x.mean(dim=2, keepdim=True) #mean with respect to the time axis
                i_std = x.std(dim=2, keepdim=True) #std with respect to the time axis
                x = (x - i_mean) / (i_std + 1e-7) #z-score standardization with respect to time axis
            encoding_indices = encoding_indices.view(-1, 1)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
            encodings.scatter_(1, encoding_indices, 1) #[9984, 16] one-hot vectors

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.embeddings.weight).view(-1, 64, 25) #[10016, 16] X [16, 64] = [10016, 64] -> reshape [32, 313, 64]

            # convert quantized from BHWC -> BCHW
            # for TS, BLC -> BCL

            if self.use_nor:
                quantized = quantized*(i_std + 1e-7) + i_mean

            return quantized #[1, 64, 103]

    def forward(self, x):
        
        if self.use_nor:
            i_mean = x.mean(dim=2, keepdim=True) #mean with respect to the time axis
            i_std = x.std(dim=2, keepdim=True) #std with respect to the time axis
            x = (x - i_mean) / (i_std + 1e-7) #z-score standardization with respect to time axis

        flat_input = x.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t())) #[9984 (32 * 64), 16]
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x) #loss between quantized (frozen) and inputs
        q_latent_loss = F.mse_loss(quantized, x.detach()) #loss between quantized and inputs (frozen) a
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        if self.use_nor:
            quantized = quantized*(i_std + 1e-7) + i_mean

        return quantized, loss, encoding_indices

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, kernel_size, stride_size, output_dim=1):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=kernel_size, stride=stride_size, padding=0, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride_size, padding=0)
        self.deconv3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0)
        self.deconv6 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=4, stride=4, padding=0)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = self.deconv6(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, commitment_cost, kernel_size, stride_size, use_nor):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim, kernel_size, stride_size)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, use_nor)
        self.decoder = Decoder(embedding_dim, hidden_dim, kernel_size, stride_size, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, encoding_indices, quantized 

    def code_decode(self, codes):
        quantized = self.quantizer.infer(codes)
        x_recon = self.decoder(quantized)
        return x_recon

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            quantized, _, encoding_indices = self.quantizer(z)
        return quantized, encoding_indices

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

 
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

        self.model = VQVAE(self.args.input_dim, self.args.hidden_dim, self.args.embedding_dim, self.args.num_embeddings, self.args.commitment_cost, self.args.kernel_size, self.args.stride_size, self.args.use_nor).to(device)
        print(self.model)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), self.args.lr, amsgrad=False)

        scheduler = MultiStepLR(optimizer, milestones=[80], gamma=0.2)
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

                data_recon, vq_loss, encoding_indices, quantized = self.model(data)
                recon_error = F.mse_loss(data_recon, data)
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
                    data_recon, vq_loss, encoding_indices, quantized = self.model(data)
                    recon_error= F.mse_loss(data_recon, data)
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
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_embeddings', type=int, default=64)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--stride_size', type=int, default=2)
    parser.add_argument('--use_nor', type=str2bool, help='', default=False)

    parser.add_argument('--savedir', type=str, default="./vqvae_model")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='train', choice=['test', 'train'])

    #if test
    parser.add_argument('--test_data', type=str, default=False, help="Quantized test data for decoding")
    parser.add_argument('--load_model', type=str, default=False, help="VQVAE model path")

    args = parser.parse_args()

    vqtrain = VQTrainer(args)
    if args.mode == "train":
        vqtrain.train()
    
    if args.mode =="test":
        vqtrain.test()



    


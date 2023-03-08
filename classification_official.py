import torch.nn as nn 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import copy
import argparse
import pickle
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torch.utils.data import Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.model_selection import train_test_split
import math 
from vqvae_official import VQVAE
from lime import explanation 
from lime import lime_base
import sklearn 
import numpy as np 
import wandb
import logging 
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()
wandb.init()
torch.manual_seed(911)

class VQVAE_Conv(nn.Module):
    def __init__(self, *, n_emb, num_classes, vqvae_model):
        super().__init__()

        #load vqvae model 
        vae_path = vqvae_model
        load_dict = torch.load(vae_path)["model_state_dict"]
        self.args= torch.load(vae_path)["args"]

        #vqvae model
        self.vae = VQVAE(self.args.input_dim, self.args.hidden_dim, self.args.embedding_dim, self.args.num_embeddings, self.args.commitment_cost, self.args.kernel_size, self.args.stride_size, self.args.use_nor).to(device)
        self.vae.load_state_dict(load_dict)

        for param in self.vae.parameters():
            param.requires_grad = False    
        
        self.to_hidden = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        self.embedding = nn.Embedding(64, 64)

        self.mlp_head = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.ReLU()
        )
    def predict(self, X):
        X = tensor(X, dtype=torch.float32)
        X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                x = self.embedding(x[0].squeeze(1).long())
                x = x.transpose(2,1)
                x = self.to_hidden(x)
                x = x.transpose(2,1)
                x = x.mean(dim=1)
                y_hat = self.mlp_head(x)
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
                
        return result

    def quantize(self, img):
        vq_x = self.vae(img)
        codebook_x = vq_x[3][0].reshape(img.shape[0], -1)
        x = vq_x[4]

        return codebook_x, x 
    
    def forward(self, img):
        vq_x = self.vae(img)
        
        x_recon = vq_x[0]
        #Codebook indices
        codebook_x = vq_x[2].reshape(img.shape[0], -1) #[bs, 312] 
        
        #Embedding of codebook
        x = self.embedding(codebook_x)
        x = x.transpose(2,1)

        x = self.to_hidden(x)

        x = x.transpose(2,1) #[64, 64, 102]
        x = x.mean(dim = 1)
        
        return self.mlp_head(x), codebook_x, x_recon 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    #Model Configuration
    parser.add_argument('--classification', type=str, default="./classification_model/")
    parser.add_argument('--vqvae_model', type=str, default="./vqvae_model/model_35330_trained_vqvae.pt")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=30000)
    parser.add_argument('--n_emb', type=int, default=64)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    net = VQVAE_Conv(
        n_emb = args.n_emb,
        num_classes = 4,
        vqvae_model = args.vqvae_model
    )

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
    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    #training
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=2e-4)

    os.makedirs(args.classification, exist_ok=True)

    #Traning the Network 
    net = net.to(device) 
    for epoch in range(args.n_epochs):
        net.train()
        
        training_loss = 0
        output_list, labels_list = [], []
        
        for _, (data, labels) in enumerate(training_loader):
            data = data.unsqueeze(1).float()
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            output, codebook, _ = net(data) 
            loss = criterion(output, labels) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * data.size(0)
            output_list.append(output.data.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
        training_loss = training_loss / len(training_loader.sampler)
        wandb.log({"Training loss": training_loss})
        
        net.eval()
        validation_loss = 0
        correct = 0
        total = 0
        for _, (data, labels) in enumerate(validation_loader):
            data = data.unsqueeze(1).float()
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            output, codebook, _ = net(data)
            _,predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(output, labels) 
        accuracy = correct / total

        validation_loss += loss.item() * data.size(0)
        validation_loss = validation_loss / len(validation_loader.sampler)
        wandb.log({"Validation loss": validation_loss, "Accuracy": accuracy})
        if epoch % 500 == 0:
            print(accuracy)
        if epoch % 500 == 0:
            savedict = {
                'args': args,
                'model_state_dict': net.state_dict(),
                    }
            savedir = args.classification
            torch.save(savedict, f"{savedir}/classification_{epoch}.pt") 



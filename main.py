import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch_geometric

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split


'--------------------------------GNN-Model-and-Dataset-----------------------------'
class MolecularDataset(torch_geometric.data.Dataset):
    def __init__(self, pos_data, type_data, fe_data):
        self.pos_data = pos_data
        self.type_data = type_data
        self.fe_data = fe_data
        
    def len(self):
        return len(self.pos_data)
    
    def get(self, idx):
        pos = torch.tensor(self.pos_data[idx], dtype=torch.float32)
        atom_types = torch.tensor(self.type_data[idx], dtype=torch.long)
        
        # One-hot encoding for atom types (1-16)
        x = torch.zeros((len(atom_types), 16), dtype=torch.float32)
        x.scatter_(1, (atom_types-1).unsqueeze(1), 1)
        
        # Create edges (fully connected)
        num_atoms = len(atom_types)
        row = torch.arange(num_atoms).repeat_interleave(num_atoms-1)
        col = torch.cat([torch.cat([torch.arange(0,i), torch.arange(i+1,num_atoms)]) 
                        for i in range(num_atoms)])
        edge_index = torch.stack([row, col], dim=0)
        
        # Edge features (distances)
        edge_attr = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
        
        return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, 
                   y=torch.tensor([self.fe_data[idx]], dtype=torch.float32))

class FormationEnergyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 128
        
        # Node embedding
        self.node_emb = nn.Linear(16, hidden_dim)
        
        # Message passing layers
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.node_emb(data.x)
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = self.conv3(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.regressor(x).squeeze(-1)

'------------------------------------Metrics------------------------------------'
def denormalize(y, mu, std):
    return y * std + mu

def mape(pred, true, mu, std):
    """Mean Absolute Percentage Error (%)"""
    pred, true = denormalize(pred, mu, std), denormalize(true, mu, std)
    return (torch.abs((true - pred) / true).mean() * 100).item()

def relative_error(pred, true, mu, std):
    """Symmetric Relative Error (%)"""
    pred, true = denormalize(pred, mu, std), denormalize(true, mu, std)
    denominator = (torch.abs(true) + torch.abs(pred) + 1e-8) / 2
    return ((torch.abs(true - pred) / denominator).mean() * 100).item()

def accuracy(pred, true, mu, std, threshold=5):
    """% of predictions within threshold% of true values"""
    pred, true = denormalize(pred, mu, std), denormalize(true, mu, std)
    errors = torch.abs((true - pred) / true) * 100
    return ((errors < threshold).float().mean() * 100).item()

'------------------------------------GNN-Trainer------------------------------------'
def GNN_trainer(model, dataloaders, optimizer, mu, std, device, num_epochs=100, 
                checkpoint_dir="checkpoints_GNN", train=True):

    train_loader, test_loader = dataloaders
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    if not train:
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"No checkpoint found at {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print(f"Loaded pretrained model from {best_model_path}")
        return model  

    best_test_mse = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_mse, train_mape, train_rel, train_acc = 0, 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            train_mape += mape(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
            train_rel += relative_error(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
            train_acc += accuracy(out, batch.y.squeeze(-1), mu_tensor, std_tensor)

        model.eval()
        test_mse, test_mape, test_rel, test_acc = 0, 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                test_mse += F.mse_loss(out, batch.y.squeeze(-1)).item()
                test_mape += mape(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
                test_rel += relative_error(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
                test_acc += accuracy(out, batch.y.squeeze(-1), mu_tensor, std_tensor)

        avg_test_mse = test_mse / len(test_loader)

        print(f'\nGNN Model - Epoch: {epoch:03d}')
        print(f'Train MSE: {train_mse/len(train_loader):.4f} | '
              f'MAPE: {train_mape/len(train_loader):.2f}% | '
              f'RelErr: {train_rel/len(train_loader):.2f}% | '
              f'Acc(5%): {train_acc/len(train_loader):.2f}%')
        print(f'Test  MSE: {avg_test_mse:.4f} | '
              f'MAPE: {test_mape/len(test_loader):.2f}% | '
              f'RelErr: {test_rel/len(test_loader):.2f}% | '
              f'Acc(5%): {test_acc/len(test_loader):.2f}%')
        print('-' * 80)

        # Save checkpoint for this epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_mse': avg_test_mse
        }, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            torch.save(model.state_dict(), best_model_path)

    return model


'--------------------------------SMILES-Model-and-Dataset--------------------------------'
class SMILESDataset(Dataset):
    def __init__(self, smiles_data, fe_data):
        self.smiles_data = smiles_data
        self.fe_data = fe_data
        
        # Build character-level vocabulary
        self.char2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2char = {0: '<pad>', 1: '<unk>'}
        
        # Create vocabulary from all characters in SMILES
        chars = set()
        for smile in smiles_data:
            chars.update(list(smile))
        
        for i, char in enumerate(chars, start=2):
            self.char2idx[char] = i
            self.idx2char[i] = char
            
        self.vocab_size = len(self.char2idx)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        smile = self.smiles_data[idx]
        token_ids = [self.char2idx.get(char, 1) for char in smile]  # 1 for <unk>
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([self.fe_data[idx]], dtype=torch.float32)

class SMILESRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # pad idx=0
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.regressor(hidden[-1]).squeeze(-1)

def collate_fn(batch):
    """Pad sequences and create masks"""
    smiles, fe = zip(*batch)
    lengths = torch.tensor([len(x) for x in smiles])
    padded_smiles = pad_sequence(smiles, batch_first=True, padding_value=0)  # pad with 0
    fe = torch.stack(fe)
    return padded_smiles, fe, lengths

'------------------------------------SMILES-Trainer------------------------------------'
def SMILES_trainer(model, train_loader, test_loader, optimizer, mu, std, device, 
                   num_epochs=100, checkpoint_dir="checkpoints_SMILES", train=True):

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    if not train:
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"No checkpoint found at {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print(f"Loaded pretrained model from {best_model_path}")
        return model

    best_test_mse = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_mse, train_mape, train_rel, train_acc = 0, 0, 0, 0

        for smiles, fe, lengths in tqdm(train_loader, desc=f"Epoch {epoch}"):
            smiles, fe = smiles.to(device), fe.to(device)
            optimizer.zero_grad()
            out = model(smiles)
            loss = F.mse_loss(out, fe.squeeze(-1))
            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            train_mape += mape(out, fe.squeeze(-1), mu_tensor, std_tensor)
            train_rel += relative_error(out, fe.squeeze(-1), mu_tensor, std_tensor)
            train_acc += accuracy(out, fe.squeeze(-1), mu_tensor, std_tensor)

        model.eval()
        test_mse, test_mape, test_rel, test_acc = 0, 0, 0, 0
        with torch.no_grad():
            for smiles, fe, lengths in test_loader:
                smiles, fe = smiles.to(device), fe.to(device)
                out = model(smiles)
                test_mse += F.mse_loss(out, fe.squeeze(-1)).item()
                test_mape += mape(out, fe.squeeze(-1), mu_tensor, std_tensor)
                test_rel += relative_error(out, fe.squeeze(-1), mu_tensor, std_tensor)
                test_acc += accuracy(out, fe.squeeze(-1), mu_tensor, std_tensor)

        avg_test_mse = test_mse / len(test_loader)
        print(f'\nSMILES Model - Epoch: {epoch:03d}')
        print(f'Train MSE: {train_mse/len(train_loader):.4f} | '
              f'MAPE: {train_mape/len(train_loader):.2f}% | '
              f'RelErr: {train_rel/len(train_loader):.2f}% | '
              f'Acc(5%): {train_acc/len(train_loader):.2f}%')
        print(f'Test  MSE: {avg_test_mse:.4f} | '
              f'MAPE: {test_mape/len(test_loader):.2f}% | '
              f'RelErr: {test_rel/len(test_loader):.2f}% | '
              f'Acc(5%): {test_acc/len(test_loader):.2f}%')
        print('-' * 80)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_mse': avg_test_mse
        }, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        # Save best model
        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            torch.save(model.state_dict(), best_model_path)

    return model





def task1():

    '-----------------------------data-processing-----------------------------'
    with open('data/pos_data.pkl', 'rb') as f:
        pos_data = pickle.load(f)
    with open('data/type_data.pkl', 'rb') as f:
        type_data = pickle.load(f)
    with open('data/smiles.pkl', 'rb') as f:
        smiles_data = pickle.load(f)

    data_split = np.load('data/data_split.npz')
    train_idxes = data_split['train_idx']
    test_idxes = data_split['test_idx']
    formation_energy = np.load('data/formation_energy.npz')
    fe = formation_energy['y']
    mu = formation_energy['mu']
    std = formation_energy['sigma']


    '-----------------------------geometric-model-----------------------------'
    dataset = MolecularDataset(pos_data, type_data, fe)
    train_dataset = [dataset.get(i) for i in train_idxes]
    test_dataset = [dataset.get(i) for i in test_idxes]

    batch_size = 64
    train_loader = torch_geometric.loader.DataLoader(train_dataset, 
                                                     batch_size=batch_size, shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, 
                                                    batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = FormationEnergyGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    flag_GNN = False
    GNN_trainer(model, 
                (train_loader, test_loader), 
                optimizer, 
                mu_tensor,
                std_tensor, 
                device,
                train=flag_GNN) 
    

    '-----------------------------SMILES-model-----------------------------'
    smiles_dataset = SMILESDataset(smiles_data, fe)
    train_smiles = [smiles_dataset[i] for i in train_idxes]
    test_smiles = [smiles_dataset[i] for i in test_idxes]

    batch_size = 64
    train_smiles_loader = DataLoader(train_smiles, batch_size=batch_size, 
                                     shuffle=True, collate_fn=collate_fn)
    test_smiles_loader = DataLoader(test_smiles, batch_size=batch_size, 
                                    collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    smiles_model = SMILESRegressor(smiles_dataset.vocab_size).to(device)
    smiles_optimizer = torch.optim.Adam(smiles_model.parameters(), lr=0.001)
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    flag_smiles = False
    SMILES_trainer(smiles_model, 
               train_smiles_loader, 
               test_smiles_loader, 
               smiles_optimizer, 
               mu_tensor, 
               std_tensor, 
               device,
               train=flag_smiles)

    
def task2():
    '-----------------------------data-processing-----------------------------'
    with open('data/pos_data.pkl', 'rb') as f:
        pos_data = pickle.load(f)
    with open('data/type_data.pkl', 'rb') as f:
        type_data = pickle.load(f)
    with open('data/smiles.pkl', 'rb') as f:
        smiles_data = pickle.load(f)

    data_split = np.load('data/data_split.npz')
    train_idxes = data_split['train_idx']
    test_idxes = data_split['test_idx']
    formation_energy = np.load('data/formation_energy.npz')
    fe = formation_energy['y']
    mu = formation_energy['mu']
    std = formation_energy['sigma']

    subset_size = 100

    '-----------------------------geometric-model-----------------------------'
    dataset = MolecularDataset(pos_data, type_data, fe)
    train_dataset = [dataset.get(i) for i in train_idxes[:subset_size]]
    test_dataset = [dataset.get(i) for i in test_idxes]

    batch_size = 64
    train_loader = torch_geometric.loader.DataLoader(train_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, 
                                                    batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = FormationEnergyGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    flag_GNN = False
    GNN_trainer(model, 
                (train_loader, test_loader), 
                optimizer, 
                mu_tensor,
                std_tensor, 
                device,
                checkpoint_dir= f"checkpoints_GNN_s{subset_size}",
                train=flag_GNN) 
    

    '-----------------------------SMILES-model-----------------------------'
    smiles_dataset = SMILESDataset(smiles_data, fe)
    train_smiles = [smiles_dataset[i] for i in train_idxes[:subset_size]]
    test_smiles = [smiles_dataset[i] for i in test_idxes]

    batch_size = 64
    train_smiles_loader = DataLoader(train_smiles, batch_size=batch_size, 
                                     shuffle=True, collate_fn=collate_fn)
    test_smiles_loader = DataLoader(test_smiles, batch_size=batch_size, 
                                    collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    smiles_model = SMILESRegressor(smiles_dataset.vocab_size).to(device)
    smiles_optimizer = torch.optim.Adam(smiles_model.parameters(), lr=0.001)
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    flag_smiles = False
    SMILES_trainer(smiles_model, 
               train_smiles_loader, 
               test_smiles_loader, 
               smiles_optimizer, 
               mu_tensor, 
               std_tensor, 
               device,
               checkpoint_dir= f"checkpoints_SMILES_s{subset_size}",
               train=flag_smiles)
    

if __name__ == '__main__':
    print("Running Task 1...")
    task1()
    print("\n" + "="*50 + "\n")
    print("Running Task 2...")
    task2()
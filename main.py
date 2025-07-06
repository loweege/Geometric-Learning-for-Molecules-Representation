import pickle
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
import torch.nn.functional as F
import os
import torch_geometric
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MolecularDataset, SMILESDataset
from models import SMILESRegressor, FormationEnergyGNN
from utils import denormalize, mape, relative_error, accuracy, r_squared
from utils import collate_fn, evaluate_and_plot_all, load_metrics, plot_metrics

'------------------------------------GNN-Trainer------------------------------------'
def GNN_trainer(model, 
                dataloaders, 
                optimizer, 
                mu, 
                std, 
                device, 
                num_epochs=10, 
                checkpoint_dir="checkpoints_GNN", 
                train=True):

    metrics_history = {
    'train_mse': [],
    'train_r2': [],
    'test_mse': [],
    'test_r2': []
    }

    train_loader, test_loader = dataloaders
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    if not train:
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"No checkpoint found at {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        print(f"Loaded pretrained model from {best_model_path}")
        return model  

    best_test_mse = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_mse, train_mae, train_mape, train_rel, train_acc, train_r2 = 0, 0, 0, 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

            train_mse += F.mse_loss(out * std_tensor + mu_tensor, batch.y.squeeze(-1) * std_tensor + mu_tensor).item()
            train_mae += F.l1_loss(denormalize(out, mu_tensor, std_tensor), denormalize(batch.y.squeeze(-1), mu_tensor, std_tensor)).item()
            train_mape += mape(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
            train_rel += relative_error(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
            train_acc += accuracy(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
            train_r2 += r_squared(out, batch.y.squeeze(-1), mu_tensor, std_tensor)

        avg_train_mse = train_mse / len(train_loader)
        avg_train_r2 = train_r2 / len(train_loader)
        metrics_history['train_mse'].append(avg_train_mse)
        metrics_history['train_r2'].append(avg_train_r2)

        model.eval()
        test_mse, test_mae, test_mape, test_rel, test_acc, test_r2 = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                test_mse += F.mse_loss(denormalize(out, mu_tensor, std_tensor), denormalize(batch.y.squeeze(-1), mu_tensor, std_tensor)).item()
                test_mae += F.l1_loss(denormalize(out, mu_tensor, std_tensor), denormalize(batch.y.squeeze(-1), mu_tensor, std_tensor)).item()
                test_mape += mape(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
                test_rel += relative_error(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
                test_acc += accuracy(out, batch.y.squeeze(-1), mu_tensor, std_tensor)
                test_r2 += r_squared(out, batch.y.squeeze(-1), mu_tensor, std_tensor)

        avg_test_mse = test_mse / len(test_loader)
        avg_test_r2 = test_r2 / len(test_loader)
        metrics_history['test_mse'].append(avg_test_mse)
        metrics_history['test_r2'].append(avg_test_r2)
        avg_test_mse = test_mse / len(test_loader)

        print(f'\nGNN Model - Epoch: {epoch:03d}')
        print(f'Train MSE: {train_mse/len(train_loader):.4f} | '
              f'Train MAE: {train_mae/len(train_loader):.4f} | '
              f'MAPE: {train_mape/len(train_loader):.2f}% | '
              f'RelErr: {train_rel/len(train_loader):.2f}% | '
              f'Acc(5%): {train_acc/len(train_loader):.2f}% | '
              f'R2: {train_r2/len(train_loader):.4f}')
        print(f'Test  MSE: {avg_test_mse:.4f} | '
              f'Test MAE: {test_mae/len(test_loader):.4f} | '
              f'MAPE: {test_mape/len(test_loader):.2f}% | '
              f'RelErr: {test_rel/len(test_loader):.2f}% | '
              f'Acc(5%): {test_acc/len(test_loader):.2f}% | '
              f'R2: {test_r2/len(test_loader):.4f}')
        print('-' * 80)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_mse': avg_test_mse
        }, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            torch.save(model.state_dict(), best_model_path)

    metrics_file = os.path.join(checkpoint_dir, "GNN_metrics_history.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics_history, f)

    return model, metrics_history


'------------------------------------SMILES-Trainer------------------------------------'
def SMILES_trainer(model, 
                   train_loader, 
                   test_loader, 
                   optimizer, 
                   mu, 
                   std, 
                   device, 
                   num_epochs=10, 
                   checkpoint_dir="checkpoints_SMILES", 
                   train=True):

    metrics_history = {
        'train_mse': [],
        'train_r2': [],
        'test_mse': [],
        'test_r2': []
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    if not train:
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"No checkpoint found at {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        print(f"Loaded pretrained model from {best_model_path}")
        return model

    best_test_mse = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_mse, train_mae, train_mape, train_rel, train_acc, train_r2 = 0, 0, 0, 0, 0, 0

        for smiles, fe, lengths in tqdm(train_loader, desc=f"Epoch {epoch}"):
            smiles, fe = smiles.to(device), fe.to(device)
            optimizer.zero_grad()
            out = model(smiles)
            loss = F.mse_loss(out, fe.squeeze(-1))
            loss.backward()
            optimizer.step()

            train_mse += F.mse_loss(denormalize(out, mu_tensor, std_tensor), denormalize(fe.squeeze(-1), mu_tensor, std_tensor)).item()
            train_mae += F.l1_loss(denormalize(out, mu_tensor, std_tensor), denormalize(fe.squeeze(-1), mu_tensor, std_tensor)).item()
            train_mape += mape(out, fe.squeeze(-1), mu_tensor, std_tensor)
            train_rel += relative_error(out, fe.squeeze(-1), mu_tensor, std_tensor)
            train_acc += accuracy(out, fe.squeeze(-1), mu_tensor, std_tensor)
            train_r2 += r_squared(out, fe.squeeze(-1), mu_tensor, std_tensor)

        avg_train_mse = train_mse / len(train_loader)
        avg_train_r2 = train_r2 / len(train_loader)
        metrics_history['train_mse'].append(avg_train_mse)
        metrics_history['train_r2'].append(avg_train_r2)

        model.eval()
        test_mse, test_mae, test_mape, test_rel, test_acc, test_r2 = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for smiles, fe, lengths in test_loader:
                smiles, fe = smiles.to(device), fe.to(device)
                out = model(smiles)

                test_mse += F.mse_loss(denormalize(out, mu_tensor, std_tensor), denormalize(fe.squeeze(-1), mu_tensor, std_tensor)).item()
                test_mae += F.l1_loss(denormalize(out, mu_tensor, std_tensor), denormalize(fe.squeeze(-1), mu_tensor, std_tensor)).item()
                test_mape += mape(out, fe.squeeze(-1), mu_tensor, std_tensor)
                test_rel += relative_error(out, fe.squeeze(-1), mu_tensor, std_tensor)
                test_acc += accuracy(out, fe.squeeze(-1), mu_tensor, std_tensor)
                test_r2 += r_squared(out, fe.squeeze(-1), mu_tensor, std_tensor)

        avg_test_mse = test_mse / len(test_loader)
        avg_test_r2 = test_r2 / len(test_loader)
        metrics_history['test_mse'].append(avg_test_mse)
        metrics_history['test_r2'].append(avg_test_r2)
        avg_test_mse = test_mse / len(test_loader)
        print(f'\nSMILES Model - Epoch: {epoch:03d}')
        print(f'Train MSE: {train_mse/len(train_loader):.4f} | '
              f'Train MAE: {train_mae/len(train_loader):.4f} | '
              f'MAPE: {train_mape/len(train_loader):.2f}% | '
              f'RelErr: {train_rel/len(train_loader):.2f}% | '
              f'Acc(5%): {train_acc/len(train_loader):.2f}% | '
              f'R2: {train_r2/len(train_loader):.4f}')
        print(f'Test  MSE: {avg_test_mse:.4f} | '
              f'Test MAE: {test_mae/len(test_loader):.4f} | '
              f'MAPE: {test_mape/len(test_loader):.2f}% | '
              f'RelErr: {test_rel/len(test_loader):.2f}% | '
              f'Acc(5%): {test_acc/len(test_loader):.2f}% | '
              f'R2: {test_r2/len(test_loader):.4f}')
        print('-' * 80)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_mse': avg_test_mse
        }, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            torch.save(model.state_dict(), best_model_path)

    metrics_file = os.path.join(checkpoint_dir, "SMILES_metrics_history.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics_history, f)

    return model, metrics_history


def training_and_comparison():

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
    print(f"Using device: {device}")

    model = FormationEnergyGNN(node_feat_dim=16, hidden_dim=64, num_layers=3).to(device)
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

    '-----------------------------visualization-----------------------------'
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)
    gnn_model = model.to(device)
    gnn_model.load_state_dict(torch.load("checkpoints_GNN/best_model.pt", map_location=device))
    smiles_model.load_state_dict(torch.load("checkpoints_SMILES/best_model.pt", map_location=device))

    data_split = np.load('data/data_split.npz')
    train_idxes = data_split['train_idx']
    test_idxes = data_split['test_idx']

    evaluate_and_plot_all(gnn_model, 
                          smiles_model,
                          train_loader, 
                          test_loader,
                          train_smiles_loader, 
                          test_smiles_loader,
                          mu, 
                          std, 
                          device)

if __name__ == '__main__':
    print("Starting Training and Comparison...")
    training_and_comparison()
    print("Training and Comparison phase completed successfully.")
    print("\n" + "="*50 + "\n")
    print("Loading and Plotting Metrics...")
    gnn_metrics, smiles_metrics = load_metrics()
    plot_metrics(gnn_metrics, smiles_metrics)
    print("Metrics loaded and plotted successfully.")
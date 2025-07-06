import torch
import matplotlib.pyplot as plt
import os
import pickle
from torch.nn.utils.rnn import pad_sequence

'------------------------------------Metrics------------------------------------'
def denormalize(y, mu, std):
    if isinstance(mu, torch.Tensor):
        mu = mu.to(y.device)
    if isinstance(std, torch.Tensor):
        std = std.to(y.device)
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

def r_squared(pred, true, mu, std):
    """R-squared (coefficient of determination)"""
    pred, true = denormalize(pred, mu, std), denormalize(true, mu, std)
    SS_res = torch.sum((true - pred)**2)
    SS_tot = torch.sum((true - torch.mean(true))**2)
    return (1 - SS_res / (SS_tot + 1e-8)).item()

'------------------------------------collate-function------------------------------------'
def collate_fn(batch):
    """Pad sequences and create masks"""
    smiles, fe = zip(*batch)
    lengths = torch.tensor([len(x) for x in smiles])
    padded_smiles = pad_sequence(smiles, batch_first=True, padding_value=0) 
    fe = torch.stack(fe)
    return padded_smiles, fe, lengths


'------------------------------------plot-function------------------------------------'
def evaluate_and_plot_all(model_gnn, model_smiles, 
                        train_loader_gnn, 
                        test_loader_gnn,
                        train_loader_smiles, 
                        test_loader_smiles, 
                        mu, 
                        std, 
                        device):
    model_gnn.eval()
    model_smiles.eval()
    mu_tensor = torch.tensor(mu, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(std, dtype=torch.float32).to(device)

    def get_predictions(model, loader, model_type='gnn'):
        preds, truths = [], []
        with torch.no_grad():
            for batch in loader:
                if model_type == 'gnn':
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    truths.extend(batch.y.squeeze(-1).tolist())
                    preds.extend(out.tolist())
                else:
                    smiles, fe, _ = batch
                    smiles, fe = smiles.to(device), fe.to(device)
                    out = model(smiles)
                    truths.extend(fe.squeeze(-1).tolist())
                    preds.extend(out.tolist())
        return (denormalize(torch.tensor(preds), mu, std).numpy(),
                denormalize(torch.tensor(truths), mu, std).numpy())


    pred_gnn_train, true_gnn_train = get_predictions(model_gnn, train_loader_gnn, 'gnn')
    pred_gnn_test, true_gnn_test = get_predictions(model_gnn, test_loader_gnn, 'gnn')
    pred_smiles_train, true_smiles_train = get_predictions(model_smiles, train_loader_smiles, 'smiles')
    pred_smiles_test, true_smiles_test = get_predictions(model_smiles, test_loader_smiles, 'smiles')


    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    lims = [min(true_gnn_train.min(), true_gnn_test.min(), true_smiles_train.min(), true_smiles_test.min()),
            max(true_gnn_train.max(), true_gnn_test.max(), true_smiles_train.max(), true_smiles_test.max())]

    axs[0, 0].scatter(true_gnn_train, pred_gnn_train, alpha=0.5, color='blue')
    axs[0, 0].plot(lims, lims, 'k--', lw=2)
    axs[0, 0].set_title("GNN - Train")
    axs[0, 0].set_xlabel("True Formation Energy")
    axs[0, 0].set_ylabel("Predicted Formation Energy")

    axs[0, 1].scatter(true_gnn_test, pred_gnn_test, alpha=0.5, color='blue')
    axs[0, 1].plot(lims, lims, 'k--', lw=2)
    axs[0, 1].set_title("GNN - Test")
    axs[0, 1].set_xlabel("True Formation Energy")
    axs[0, 1].set_ylabel("Predicted Formation Energy")

    axs[1, 0].scatter(true_smiles_train, pred_smiles_train, alpha=0.5, color='green')
    axs[1, 0].plot(lims, lims, 'k--', lw=2)
    axs[1, 0].set_title("SMILES - Train")
    axs[1, 0].set_xlabel("True Formation Energy")
    axs[1, 0].set_ylabel("Predicted Formation Energy")

    axs[1, 1].scatter(true_smiles_test, pred_smiles_test, alpha=0.5, color='green')
    axs[1, 1].plot(lims, lims, 'k--', lw=2)
    axs[1, 1].set_title("SMILES - Test")
    axs[1, 1].set_xlabel("True Formation Energy")
    axs[1, 1].set_ylabel("Predicted Formation Energy")

    plt.tight_layout()
    plt.suptitle("Predicted vs. True Formation Energy (GNN vs SMILES)", fontsize=16, y=1.03)
    plt.show()


'------------------------------------load-and-plot-metrics------------------------------------'
def load_metrics(base_dir="."):
    """Load all metrics files from the directory structure"""
    gnn_metrics = {}
    smiles_metrics = {}
    
    for dirname in os.listdir(base_dir):
        if dirname.startswith("checkpoints_GNN_s"):
            size = int(dirname.split("_s")[1])
            metrics_file = os.path.join(base_dir, dirname, "GNN_metrics_history.pkl")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'rb') as f:
                    gnn_metrics[size] = pickle.load(f)

    for dirname in os.listdir(base_dir):
        if dirname.startswith("checkpoints_SMILES_s"):
            size = int(dirname.split("_s")[1])
            metrics_file = os.path.join(base_dir, dirname, "SMILES_metrics_history.pkl")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'rb') as f:
                    smiles_metrics[size] = pickle.load(f)
    
    return gnn_metrics, smiles_metrics

def plot_metrics(gnn_metrics, smiles_metrics):
    plt.figure(figsize=(20, 16))
    gnn_sizes = sorted(gnn_metrics.keys())
    smiles_sizes = sorted(smiles_metrics.keys())
    
    # ====================== MSE PLOTS ======================
    plt.subplot(2, 4, 1)
    for size in gnn_sizes:
        metrics = gnn_metrics[size]
        epochs = range(1, len(metrics['train_mse']) + 1)
        plt.plot(epochs, metrics['train_mse'], label=f'{size} samples', alpha=0.8)
    plt.title('GNN Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    

    plt.subplot(2, 4, 2)
    for size in gnn_sizes:
        metrics = gnn_metrics[size]
        epochs = range(1, len(metrics['test_mse']) + 1)
        plt.plot(epochs, metrics['test_mse'], label=f'{size} samples', alpha=0.8)
    plt.title('GNN Testing MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    

    plt.subplot(2, 4, 3)
    for size in smiles_sizes:
        metrics = smiles_metrics[size]
        epochs = range(1, len(metrics['train_mse']) + 1)
        plt.plot(epochs, metrics['train_mse'], label=f'{size} samples', alpha=0.8)
    plt.title('SMILES Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    

    plt.subplot(2, 4, 4)
    for size in smiles_sizes:
        metrics = smiles_metrics[size]
        epochs = range(1, len(metrics['test_mse']) + 1)
        plt.plot(epochs, metrics['test_mse'], label=f'{size} samples', alpha=0.8)
    plt.title('SMILES Testing MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    
    # ====================== R² PLOTS ======================
    plt.subplot(2, 4, 5)
    for size in gnn_sizes:
        metrics = gnn_metrics[size]
        epochs = range(1, len(metrics['train_r2']) + 1)
        plt.plot(epochs, metrics['train_r2'], label=f'{size} samples', alpha=0.8)
    plt.title('GNN Training R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)


    plt.subplot(2, 4, 6)
    for size in gnn_sizes:
        metrics = gnn_metrics[size]
        epochs = range(1, len(metrics['test_r2']) + 1)
        plt.plot(epochs, metrics['test_r2'], label=f'{size} samples', alpha=0.8)
    plt.title('GNN Testing R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    

    plt.subplot(2, 4, 7)
    for size in smiles_sizes:
        metrics = smiles_metrics[size]
        epochs = range(1, len(metrics['train_r2']) + 1)
        plt.plot(epochs, metrics['train_r2'], label=f'{size} samples', alpha=0.8)
    plt.title('SMILES Training R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    

    plt.subplot(2, 4, 8)
    for size in smiles_sizes:
        metrics = smiles_metrics[size]
        epochs = range(1, len(metrics['test_r2']) + 1)
        plt.plot(epochs, metrics['test_r2'], label=f'{size} samples', alpha=0.8)
    plt.title('SMILES Testing R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend(title="Training Size")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_size_comparison_with_R2.png', dpi=300, bbox_inches='tight')
    plt.show()

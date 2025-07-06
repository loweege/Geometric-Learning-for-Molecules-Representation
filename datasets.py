import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

'--------------------------------GNN-Dataset------------------------------------'   
class MolecularDataset(torch_geometric.data.Dataset):
    def __init__(self, pos_data, type_data, fe_data):
        self.pos_data = pos_data
        self.type_data = type_data
        self.fe_data = fe_data
        
    def __len__(self):
        return len(self.pos_data)
    
    def create_edges_distance_threshold(self, pos, threshold=2.0):
        """Create edges based on distance threshold"""
        num_atoms = len(pos)
        distances = squareform(pdist(pos))
        
        i_indices, j_indices = np.where((distances < threshold) & (distances > 0))
            
        edge_index = torch.stack([
            torch.tensor(i_indices, dtype=torch.long),
            torch.tensor(j_indices, dtype=torch.long)
        ], dim=0)
        
        edge_attr = torch.tensor(distances[i_indices, j_indices], dtype=torch.float32).unsqueeze(1)
        return edge_index, edge_attr
    
    def get(self, idx):
        pos = torch.tensor(self.pos_data[idx], dtype=torch.float32)
        atom_types = torch.tensor(self.type_data[idx], dtype=torch.long)
        
        x = torch.zeros((len(atom_types), 16), dtype=torch.float32)
        x.scatter_(1, (atom_types-1).unsqueeze(1), 1)
        edge_index, edge_attr = self.create_edges_distance_threshold(pos.numpy())
        
        return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, 
                   y=torch.tensor([self.fe_data[idx]], dtype=torch.float32),
                   atom_types=atom_types)


'--------------------------------SMILES-Dataset------------------------------------'   
class SMILESDataset(Dataset):
    def __init__(self, smiles_data, fe_data):
        self.smiles_data = smiles_data
        self.fe_data = fe_data
        self.token2idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        self.idx2token = {0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>'}
        tokens = set()

        for smile in smiles_data:
            tokens.update(self._tokenize_smiles(smile))
        
        for i, token in enumerate(tokens, start=4):
            self.token2idx[token] = i
            self.idx2token[i] = token
            
        self.vocab_size = len(self.token2idx)
    
    def _tokenize_smiles(self, smiles):
        """Simple atom-level tokenization"""
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':
                j = i + 1
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                tokens.append(smiles[i:j+1])
                i = j + 1
            elif smiles[i].isupper():
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    tokens.append(smiles[i:i+2])
                    i += 2
                else:
                    tokens.append(smiles[i])
                    i += 1
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens

    def __getitem__(self, idx):
        smile = self.smiles_data[idx]
        tokens = self._tokenize_smiles(smile)
        token_ids = [self.token2idx.get(token, 1) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([self.fe_data[idx]], dtype=torch.float32)
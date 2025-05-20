import torch.utils.data as data
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein


class DTIDataset(data.Dataset):
    def __init__(self, list_ids, df, max_drug_nodes=290, max_protein_length=1200):
        self.list_ids = list_ids
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_length = max_protein_length
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        index = self.list_ids[index]

        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p, self.max_protein_length)
        y = self.df.iloc[index]["Y"]
        return v_d, v_p, y

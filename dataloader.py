import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein

from rdkit import Chem
from rdkit.Chem import AllChem
from pubchemfp import GetPubChemFPs

def extract_fingerprints(smiles):
    fingerprints = []
    mol = Chem.MolFromSmiles(smiles)
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    fp_pubcfp = GetPubChemFPs(mol)

    fingerprints.extend(fp_maccs)
    fingerprints.extend(fp_phaErGfp)
    fingerprints.extend(fp_pubcfp)

    return fingerprints

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']

        v_d_graph = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d_graph.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d_graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d_graph = v_d_graph.add_self_loop()

        v_d_fingerprint = extract_fingerprints(v_d)
        v_d_fingerprint = torch.Tensor(v_d_fingerprint)

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        # y = torch.Tensor([y])
        return v_d_graph, v_d_fingerprint, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches

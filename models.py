import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from torch.nn.utils.weight_norm import weight_norm
from cross import IterativeCrossModalFeatureEnhancement

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class MFI(nn.Module):
    def __init__(self, **config):
        super(MFI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]

        fingerprint_dim = config["DRUG"]["FINGERPRINT_DIM"]
        hidden_dim = config["DRUG"]["HIDDEN_DIM"]
        output_dim = config["DRUG"]["OUTPUT_DIM"]
        dropout = config["DRUG"]["DROPOUT"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.fingerprint_ann = FingerprintANN(fingerprint_dim, hidden_dim, output_dim, dropout)
        combined_feat_dim = drug_hidden_feats[-1] * 2
        self.combined_processing = nn.Sequential(
            nn.Linear(combined_feat_dim, combined_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_feat_dim // 2, drug_hidden_feats[-1]),
            nn.ReLU()
        )
        self.protein_extractor = EMSC(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(290)
        self.cross_attention = IterativeCrossModalFeatureEnhancement(
            v_dim=128 ,
            q_dim = 128  ,
            h_dim = 256 ,
            h_out = 128 ,
            num_heads = 8 ,
            num_iterations = 1
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_d_fingerprint, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_fingerprint = self.fingerprint_ann(v_d_fingerprint)
        v_fingerprint = v_fingerprint.unsqueeze(1)
        v_fingerprint = v_fingerprint.repeat(1, v_d.size(1), 1)
        v_drug_combined = torch.cat((v_d, v_fingerprint), dim=-1)
        v_d = self.combined_processing(v_drug_combined)
        v_p = self.protein_extractor(v_p)
        f, att = self.cross_attention(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class EMSC(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, padding=True, num_heads=2, dropout=0.1):
        super(EMSC, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        in_ch = embedding_dim
        self.num_filters = num_filters

        self.conv1_3 = nn.Conv1d(in_channels=in_ch, out_channels=num_filters[0] * 2, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv1_5 = nn.Conv1d(in_channels=in_ch, out_channels=num_filters[1] * 2, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv1_7 = nn.Conv1d(in_channels=in_ch, out_channels=num_filters[2] * 2, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)

        self.bn1 = nn.BatchNorm1d(sum(num_filters))

        self.dwconv2 = nn.Conv1d(in_channels=sum(num_filters), out_channels=sum(num_filters), kernel_size=3, padding=1, groups=sum(num_filters))
        self.pointwise_conv2 = nn.Conv1d(in_channels=sum(num_filters), out_channels=num_filters[0], kernel_size=1)
        self.bn2 = nn.BatchNorm1d(num_filters[0])

        self.self_attention = nn.MultiheadAttention(embed_dim=num_filters[0], num_heads=num_heads, dropout=dropout)

    def forward(self, v):

        v = self.embedding(v.long())
        v = v.transpose(2, 1)

        v1_1, v1_2 = self.conv1_3(v).chunk(2, dim=1)
        v1 = F.gelu(v1_1) * torch.sigmoid(v1_2)

        v2_1, v2_2 = self.conv1_5(v).chunk(2, dim=1)
        v2 = F.gelu(v2_1) * torch.sigmoid(v2_2)

        v3_1, v3_2 = self.conv1_7(v).chunk(2, dim=1)
        v3 = F.gelu(v3_1) * torch.sigmoid(v3_2)

        min_len = min(v1.size(2), v2.size(2), v3.size(2))
        v1 = v1[:, :, :min_len]
        v2 = v2[:, :, :min_len]
        v3 = v3[:, :, :min_len]

        v = torch.cat((v1, v2, v3), dim=1)

        v = self.bn1(v)

        v = F.relu(self.dwconv2(v))
        v = F.relu(self.pointwise_conv2(v))
        v = self.bn2(v)

        v = v.permute(2, 0, 1)
        v, _ = self.self_attention(v, v, v)
        v = v.permute(1, 2, 0)

        v = v.transpose(2, 1)
        v = v.view(v.size(0), v.size(1), -1)

        return v

class FingerprintANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(FingerprintANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, fp):
        v = self.fc1(fp)
        v = self.act_func(v)
        v = self.dropout(v)
        v = self.fc2(v)
        return v
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x



class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

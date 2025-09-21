import torch
from torch import nn
from dgllife.model.gnn import GCN
import torch.nn.functional as F
import math
from utils import to_3d, to_4d
from HR import HR
from Top_k import TransformerBlock

class HRDS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.drug_extractor = MoleculeGCN(configs)
        self.prot_extractor = MKCNN(configs)
        self.fusion = fusion(configs)
        self.predict_dti = DropoutMLP(configs)
        self.hr = HR(
            dim=256,
            head_num=8,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            qkv_bias=True,
            fuse_bn=False,
            down_sample_mode='avg_pool',
            attn_drop_ratio=0.1,
            gate_layer='sigmoid'
        )

    def forward(self, d_graph, p_feat, mode='train'):
        v_d = self.drug_extractor(d_graph) 
        v_p = self.prot_extractor(p_feat)  
        v_d = v_d + to_3d(self.hr(to_4d(v_d, v_d.size(1), 1)))
        v_p = v_p + to_3d(self.hr(to_4d(v_p, v_p.size(1), 1)))
        f, attn = self.fusion(v_d, v_p)
        score = self.predict_dti(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, attn, score


class MoleculeGCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.in_feat = configs.Drug.Node_In_Feat
        self.dim_embedding = configs.Drug.Node_In_Embedding
        self.hidden_feats = configs.Drug.Hidden_Layers
        self.padding = configs.Drug.Padding
        self.activation = configs.Drug.GCN_Activation

        self.init_linear = nn.Linear(self.in_feat, self.dim_embedding, bias=False)
        if self.padding:
            with torch.no_grad():
                self.init_linear.weight[-1].fill_(0)
        self.gcn = GCN(in_feats=self.dim_embedding, hidden_feats=self.hidden_feats, activation=self.activation)
        self.output_feats = self.hidden_feats[-1]

    def forward(self, batch_d_graph):
        node_feats = batch_d_graph.ndata['h']
        node_feats = self.init_linear(node_feats)
        node_feats = self.gcn(batch_d_graph, node_feats)
        batch_size = batch_d_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class MKCNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embedding_dim = configs.Protein.Embedding_Dim
        self.num_filters = configs.Protein.Num_Filters
        self.kernel_size = configs.Protein.Kernel_Size
        self.padding = configs.Protein.Padding

        if self.padding:
            self.embedding = nn.Embedding(26, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, self.embedding_dim)
        in_out_ch = [self.embedding_dim] + self.num_filters
        kernels = self.kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_out_ch[0], out_channels=in_out_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_out_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_out_ch[1], out_channels=in_out_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_out_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_out_ch[2], out_channels=in_out_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_out_ch[3])

    def forward(self, p_feat):
        p_feat = self.embedding(p_feat.long())
        p_feat = p_feat.transpose(2, 1)
        p_feat = F.relu(self.bn1(self.conv1(p_feat)))
        p_feat = F.relu(self.bn2(self.conv2(p_feat)))
        p_feat = F.relu(self.bn3(self.conv3(p_feat)))
        p_feat = p_feat.transpose(2, 1)
        return p_feat




class fusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.positional_drug = PositionalEncoding(configs.fusion.Hidden_Dim, max_len=configs.Drug.Nodes)
        self.positional_prot = PositionalEncoding(configs.fusion.Hidden_Dim, max_len=configs.Protein.CNN_Length)
        self.bca = AttenMapNHeads(configs)
        self.attention_fc_dp = nn.Linear(configs.fusion.Num_Heads, configs.fusion.Hidden_Dim)
        self.attention_fc_pd = nn.Linear(configs.fusion.Num_Heads, configs.fusion.Hidden_Dim)
        self.Top_k = TransformerBlock(dim=256, num_heads=8, bias=True, LayerNorm_type='WithBias')

    def forward(self, drug, protein):
        drug = self.positional_drug(drug)
        protein = self.positional_prot(protein)
        drug = self.Top_k(drug)
        protein = self.Top_k(protein)
        attn_map = self.bca(drug, protein)
        att_dp = F.softmax(attn_map, dim=-1)  
        att_pd = F.softmax(attn_map, dim=-2) 
        attn_matrix = 0.5 * att_dp + 0.5 * att_pd  
        drug_attn = self.attention_fc_dp(torch.mean(attn_matrix, -1).transpose(-1, -2))  # [bs, d_len, nheads]
        protein_attn = self.attention_fc_pd(torch.mean(attn_matrix, -2).transpose(-1, -2))  # [bs, p_len, nheads]
        drug_attn = F.sigmoid(drug_attn)
        protein_attn = F.sigmoid(protein_attn)
        drug = drug + drug * drug_attn
        protein = protein + protein * protein_attn
        drug, _ = torch.max(drug, 1)
        protein, _ = torch.max(protein, 1)
        pair = torch.cat([drug, protein], dim=1)
        return pair, (drug_attn, protein_attn)


class DropoutMLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(configs.MLP.In_Dim * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, configs.MLP.Binary)





    def forward(self, pair):
        pair = self.dropout1(pair)
        fully1 = F.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = F.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = F.leaky_relu(self.fc3(fully2))
        pred = self.out(fully3)
        return pred


class AttenMapNHeads(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.hid_dim = configs.fusion.Hidden_Dim
        self.n_heads = configs.fusion.Num_Heads

        assert self.hid_dim % self.n_heads == 0

        self.f_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.f_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.d_k = self.hid_dim // self.n_heads

    def forward(self, d, p):
        batch_size = d.shape[0]

        Q = self.f_q(d)
        K = self.f_k(p)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return attn_weights


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

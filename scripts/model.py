import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch.nn import Linear

class RegressionRefinerV3(nn.Module):
    def __init__(self, input_dim, reg_dropout=0.2):

        super().__init__()
        self.lin1 = Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.lin2 = Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)   
        self.lin3 = Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.lin4 = Linear(512, 256)
        self.head = Linear(256, 1)
        self.dropout = nn.Dropout(reg_dropout)
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.gelu(self.ln1(self.lin1(x)))
        x = self.dropout(x)   
        

        identity = x
        out = self.gelu(self.ln2(self.lin2(x)))
        out = self.dropout(out)
        x = identity + out 
        

        x = self.gelu(self.ln3(self.lin3(x)))
        x = self.gelu(self.lin4(x))
        return self.head(x)

class GAT_MultiTask_V3(nn.Module):

    def __init__(self, atom_dim=46, fp_dim=2235, num_targets=8,

                 num_experts=2, reg_dropout=0.05, dropout_cls=0.5):

        super().__init__()

        self.hidden, self.heads = 256, 5
        self.num_targets = num_targets
        self.exp_h = 512
        self.temperature = 1.5


        self.conv_shared = GATConv(atom_dim, self.hidden, heads=self.heads, concat=False)
        self.ln_shared = nn.LayerNorm(self.hidden)    
        self.conv_shared_2 = GATConv(self.hidden, self.hidden, heads=self.heads, concat=False)
        self.ln_shared_2 = nn.LayerNorm(self.hidden)
  

        self.conv_reg_private = GATConv(self.hidden, self.hidden, heads=self.heads, concat=False)
        self.reg_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            Linear(self.hidden, self.hidden), nn.Tanh(), Linear(self.hidden, 1)
        ))

        self.target_pools = nn.ModuleList([
            AttentionalAggregation(gate_nn=nn.Sequential(
                Linear(self.hidden, self.hidden), nn.ReLU(), Linear(self.hidden, 1)
            )) for _ in range(num_targets)
        ])

        fusion_dim = self.hidden + fp_dim

        self.fc_reg = RegressionRefinerV3(fusion_dim, reg_dropout)


        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                Linear(fusion_dim, self.exp_h),
                nn.GELU(),
                nn.Dropout(0.2)
            ) for _ in range(num_experts)
        ])

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    Linear(fusion_dim, self.exp_h),
                    nn.GELU(),
                   nn.Dropout(0.2)
                ) for _ in range(num_experts)
            ]) for _ in range(num_targets)
        ])

        self.target_gates = nn.ModuleList([
            Linear(fusion_dim, num_experts * 2)   
            for _ in range(num_targets)
        ])

        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_cls),
                Linear(self.exp_h, 1)
            ) for _ in range(num_targets)
        ])

    def forward(self, data):

        x = F.elu(self.ln_shared(self.conv_shared(data.x, data.edge_index)))
        x_base = F.elu(self.ln_shared_2(self.conv_shared_2(x, data.edge_index))) + x


        x_reg_refined = F.elu(self.conv_reg_private(x_base, data.edge_index))
        reg_weights = torch.sigmoid(self.reg_pool.gate_nn(x_reg_refined).squeeze(-1))
        x_reg_graph = self.reg_pool(x_reg_refined, index=data.batch)    
        feat_reg = torch.cat([x_reg_graph, data.mol_feature], dim=1)
        mic_pred = self.fc_reg(feat_reg)



        all_logits = []
        all_cls_weights = [] 
        all_gate_weights = []
        for i in range(self.num_targets):

            target_w = torch.sigmoid(self.target_pools[i].gate_nn(x_base).squeeze(-1))
            all_cls_weights.append(target_w)
            x_cls_graph = self.target_pools[i](x_base, index=data.batch)
            feat_cls_in = torch.cat([x_cls_graph, data.mol_feature], dim=1)
            shared_out = torch.stack([exp(feat_cls_in) for exp in self.shared_experts], dim=1)
            task_out = torch.stack(
                [exp(feat_cls_in) for exp in self.task_experts[i]], dim=1
            )

            all_experts = torch.cat([shared_out, task_out], dim=1)

            gate_logits = self.target_gates[i](feat_cls_in) / self.temperature
            gate_w = F.softmax(gate_logits, dim=-1).unsqueeze(1)
            combined_cls_feat = torch.bmm(gate_w, all_experts).squeeze(1)
            all_logits.append(self.target_heads[i](combined_cls_feat))
            all_gate_weights.append(gate_w.squeeze(1))

        act_logit = torch.cat(all_logits, dim=1)
        return mic_pred, act_logit, reg_weights, all_cls_weights, all_gate_weights

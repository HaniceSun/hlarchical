import os
import pandas as pd
import torch
import torch.nn as nn

class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_channels=2,
        input_length=1000,
        hidden_dims=(128, 64),
        dropout=0.3,
    ):
        super().__init__()

        self.input_dim = input_channels * input_length

        layers = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)


class HierarchicalHLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.moe = False
        self.masks = {}
        if hasattr(cfg, 'masks_file'):
            if os.path.exists(cfg.masks_file):
                df = pd.read_table(cfg.masks_file, header=0, sep='\t')
                for n in range(df.shape[0]):
                    mask = df.iloc[n, 1:].values.astype(bool)
                    self.masks[df.iloc[n, 0]] = mask
                    cfg.input_length = len(mask)
            else:
                print(f'no masks file found {cfg.masks_file}')
        else:
            print('input_length needed when masks file not provided')

        if hasattr(cfg, 'moe') and cfg.moe:
            self.moe = True
            print(f'using Mixture of Experts with masks from {cfg.masks_file}')

        self.maps = {}
        self.expert_to_head = {}
        if hasattr(cfg, 'maps_file'):
            if os.path.exists(cfg.maps_file):
                df = pd.read_table(cfg.maps_file)
                for n in range(df.shape[0]):
                    head = df['head'].iloc[n]
                    expert = df['expert'].iloc[n]
                    label = df['label'].iloc[n]
                    self.maps[head] = [label, expert]
                    self.expert_to_head[expert] = head
            else:
                print(f'no maps file found {cfg.maps_file}')

        backbone_class = eval(cfg.backbone_class)
        if not self.moe:
            self.backbone = backbone_class(
                input_channels=cfg.input_channels,
                input_length=cfg.input_length,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
            )
            self.heads = nn.ModuleDict({head:nn.Linear(self.backbone.output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})
        else:
            self.experts = nn.ModuleDict()
            for e in self.expert_to_head:
                mask = self.masks[e]
                expert = backbone_class(
                    input_channels=cfg.input_channels,
                    input_length=sum(mask),
                    hidden_dims=cfg.hidden_dims,
                    dropout=cfg.dropout,
                )
                self.experts[e] = expert
            self.heads = nn.ModuleDict({head:nn.Linear(self.experts[self.maps[head][-1]].output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})

    def forward(self, x):
        if not self.moe:
            x = self.backbone(x)
            outputs = {}
            for head in self.heads:
                h = self.heads[head](x)
                h = h.view(h.size(0), -1, 2)
                outputs[head] = h
            return outputs
        else:
            experts = {}
            for e in self.experts:
                mask = self.masks[e]
                x_m = x[:, :, mask]
                x_out = self.experts[e](x_m)
                experts[e] = x_out

            outputs = {}
            for head in self.heads:
                e = self.maps[head][-1]
                x_e = experts[e]
                h = self.heads[head](x_e)
                h = h.view(h.size(0), -1, 2)
                outputs[head] = h
            return outputs

if __name__ == "__main__":
    class Config:
        pass

    cfg = Config()
    cfg.input_channels = 2
    cfg.input_length = 1000
    cfg.hidden_dims = (128, 64)
    cfg.dropout = 0.3
    cfg.maps_file = 'maps.txt'
    cfg.moe = True
    cfg.masks_file = 'masks.txt'
    cfg.backbone_class = MLPBackbone

    model = HierarchicalHLA(cfg)
    data = torch.randn(4, 2, 1000)
    outputs = model(data)
    #print(outputs['HLA-A'].shape)
    #print(outputs['HLA-B'].shape)

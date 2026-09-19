"""Neural network backbones and the hierarchical multi-head HLA classification model."""

from typing import Any

import torch
import torch.nn as nn
from .utils import *


class MLPBackbone(nn.Module):
    """Fully-connected backbone that flattens a multi-channel 1D input and encodes it.

    Each hidden layer is a ``Linear -> ReLU -> BatchNorm1d -> Dropout`` block.

    Attributes:
        input_dim: Flattened input dimensionality (``input_channels * input_length``).
        net: The stacked ``nn.Sequential`` of hidden linear blocks.
        output_dim: Dimensionality of the final hidden layer's output.
    """

    def __init__(
        self,
        input_channels: int = 2,
        input_length: int = 1000,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        """Initializes the MLP backbone.

        Args:
            input_channels: Number of input channels.
            input_length: Length of the input sequence along the last dimension.
            hidden_dims: Sizes of successive hidden linear layers.
            dropout: Dropout probability applied after each hidden layer.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens and encodes the input.

        Args:
            x: Input tensor of shape ``(batch, input_channels, input_length)``.

        Returns:
            Encoded tensor of shape ``(batch, output_dim)``.
        """
        x = x.flatten(start_dim=1)
        return self.net(x)


class CNNBackbone(nn.Module):
    """1D convolutional backbone with stacked conv blocks and global pooling.

    Each stage is a ``Conv1d`` optionally followed by ``BatchNorm1d``, then
    ``ReLU`` and ``Dropout``. The final feature map is global-pooled (average
    or max) down to a single value per channel.

    Attributes:
        conv: The stacked ``nn.Sequential`` of convolutional blocks.
        pool: The global pooling layer (``AdaptiveAvgPool1d`` or ``AdaptiveMaxPool1d``).
        output_dim: Number of output channels, equal to ``hidden_dims[-1]``.
    """

    def __init__(
        self,
        input_channels: int = 2,
        input_length: int = 1000,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        strides: tuple[int, ...] = (1, 1, 1),
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        global_pool: str = "avg",  # "avg" or "max"
    ) -> None:
        """Initializes the CNN backbone.

        Args:
            input_channels: Number of input channels.
            input_length: Length of the input sequence (not used directly by
                this constructor; kept for interface symmetry with other
                backbones).
            hidden_dims: Number of output channels for each conv stage.
            kernel_sizes: Kernel size for each conv stage.
            strides: Stride for each conv stage.
            dropout: Dropout probability applied after each conv stage.
            use_batchnorm: Whether to apply ``BatchNorm1d`` after each conv.
            global_pool: Pooling strategy, either ``"avg"`` or ``"max"``.

        Raises:
            ValueError: If `global_pool` is not `"avg"` or `"max"`.
        """
        super().__init__()

        assert len(hidden_dims) == len(kernel_sizes) == len(strides)

        layers = []
        prev_c = input_channels

        for out_c, k, s in zip(hidden_dims, kernel_sizes, strides):
            layers.append(
                nn.Conv1d(
                    prev_c,
                    out_c,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_c = out_c

        self.conv = nn.Sequential(*layers)

        if global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input through the conv stack and global pooling.

        Args:
            x: Input tensor of shape ``(batch, channels, length)``.

        Returns:
            Pooled feature tensor of shape ``(batch, output_dim)``.
        """
        x = self.conv(x)
        x = self.pool(x)      # (B, C, 1)
        x = x.squeeze(-1)     # (B, C)
        return x


class ResidualBlock(nn.Module):
    """Pre-activation 1D residual convolutional block.

    Applies ``BatchNorm -> ReLU -> Conv1d`` twice with a shared dilation rate,
    then adds the block's input back as a residual (skip) connection.

    Attributes:
        bn1: First batch-norm layer.
        bn2: Second batch-norm layer.
        conv1: First dilated 1D convolution.
        conv2: Second dilated 1D convolution.
    """

    def __init__(self, N: int, W: int, D: int) -> None:
        """Initializes the residual block.

        Args:
            N: Number of input/output channels (preserved through the block).
            W: Convolution kernel width.
            D: Dilation rate applied to both convolutions.
        """
        super().__init__()
        self.bn1 = nn.BatchNorm1d(N)
        self.bn2 = nn.BatchNorm1d(N)
        self.conv1 = nn.Conv1d(N, N, W, dilation=D, padding='same')
        self.conv2 = nn.Conv1d(N, N, W, dilation=D, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the pre-activation residual transform.

        Args:
            x: Input tensor of shape ``(batch, N, length)``.

        Returns:
            Output tensor of the same shape as `x`, with the input added back
            as a residual connection.
        """
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class SpliceAIBackbone(torch.nn.Module):
    """SpliceAI-style dilated residual-convolution backbone with skip connections.

    Stacks `ResidualBlock`s with kernel widths and dilation rates taken from
    `cfg.NWD`, periodically accumulating a 1x1-conv projection of the running
    activation into a skip-connection sum. The skip sum is global-pooled to
    produce a fixed-size embedding.

    Attributes:
        cfg: The configuration object supplied at construction.
        conv1: Initial 1x1 conv projecting the input to `cfg.out_channels`.
        conv2: 1x1 conv used to seed the skip-connection accumulator.
        resblocks: `nn.ModuleList` of `ResidualBlock`s, one per entry in `cfg.NWD`.
        convs: `nn.ModuleList` of 1x1 convs added to the skip connection every
            `cfg.n_blocks` residual blocks.
        pool: Global pooling layer (`AdaptiveAvgPool1d` or `AdaptiveMaxPool1d`).
        output_dim: Number of output channels, equal to `cfg.out_channels`.
    """

    def __init__(self, cfg: Any) -> None:
        """Initializes the backbone from a config object.

        Args:
            cfg: Namespace-like config object (see `hlarchical.utils.Config`)
                expected to provide: `in_channels` (int), `out_channels`
                (int), `NWD` (sequence of `(N, W, D)` tuples, one per
                residual block), `n_blocks` (int, period at which a
                skip-projection conv is added), and `global_pool` (`"avg"` or
                `"max"`).

        Raises:
            ValueError: If `cfg.global_pool` is not `"avg"` or `"max"`.
        """
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.in_channels, cfg.out_channels, 1)
        self.conv2 = nn.Conv1d(cfg.out_channels, cfg.out_channels, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(cfg.NWD)):
            n, w, d = cfg.NWD[i]
            self.resblocks.append(ResidualBlock(n, w, d))
            if (i + 1) % cfg.n_blocks == 0:
                self.convs.append(nn.Conv1d(cfg.out_channels, cfg.out_channels, 1))

        if cfg.global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif cfg.global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")
        self.output_dim = cfg.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input through the dilated residual stack.

        Args:
            x: Input tensor of shape `(batch, cfg.in_channels, length)`.

        Returns:
            Pooled feature tensor of shape `(batch, cfg.out_channels)`.

        The local skip-projection index `j` is reset to `0` at the top of
        every loop iteration, so only `self.convs[0]` is ever used to update
        the skip connection even when multiple skip-projection convs exist.
        This reflects current behavior and is left unchanged here.
        """
        out = self.conv1(x)
        skip = self.conv2(out)
        for i in range(len(self.cfg.NWD)):
            n, w, d = self.cfg.NWD[i]
            out = self.resblocks[i](out)
            j = 0
            if (i + 1) % self.cfg.n_blocks == 0:
                cv = self.convs[j](out)
                skip = cv + skip
                j += 1
        out = self.pool(skip)
        out = out.squeeze(-1)
        return out


class HierarchicalHLA(nn.Module):
    """Hierarchical multi-head HLA allele classifier.

    Wraps a shared or per-expert backbone (MLP, CNN, or SpliceAI-style) with
    one linear classification head per HLA "head" defined in `maps_file`
    (e.g. one head per digit-resolution level of each HLA gene). When
    `cfg.moe` is set, a separate backbone ("expert") is instantiated per
    expert group (as defined by `masks_file`), with each head reading from
    the output of its assigned expert.

    Attributes:
        moe: Whether Mixture-of-Experts (per-expert backbones) is enabled.
        masks: Mapping of expert name to a boolean feature mask (as read from
            `masks_file`); empty if `masks_file` is `None`.
        maps: Mapping of head name to `[label, expert]`, where `label` is the
            maximum class label (number of classes minus one) and `expert` is
            the name of the expert group backing this head.
        expert_to_head: Mapping of expert name to head name, as read from
            `maps_file`.
        backbone: The single shared backbone module (only set when `moe` is
            `False`).
        experts: `nn.ModuleDict` mapping expert name to its backbone module
            (only set when `moe` is `True`).
        heads: `nn.ModuleDict` mapping head name to its linear classification
            head.
    """

    def __init__(
        self,
        cfg: Any,
        maps_file: str = 'maps.txt',
        masks_file: str | None = 'masks.txt',
    ) -> None:
        """Builds the backbone(s) and per-head classification layers.

        Args:
            cfg: Namespace-like config object (see `hlarchical.utils.Config`)
                expected to provide `backbone` (`"mlp"`, `"cnn"`, or
                `"spliceai"`), the corresponding backbone hyperparameters
                (e.g. `input_channels`, `hidden_dims`, `dropout`, ...), and
                optionally `moe` (bool) to enable Mixture-of-Experts. If
                `masks_file` is `None`, `cfg.input_length` must already be set.
            maps_file: Path to a tab-separated file with columns `head`,
                `expert`, `label` describing each classification head.
            masks_file: Path to a tab-separated file with a boolean
                feature-mask row per expert, used to size per-expert
                backbones and to derive `cfg.input_length`. If `None`,
                `cfg.input_length` is used directly and no expert feature
                masking is available.

        Raises:
            ValueError: If `masks_file` is `None` and `cfg` has no
                `input_length` attribute.
        """
        super().__init__()
        self.moe = False
        self.masks = {}

        if masks_file is not None:
            df = pd.read_table(masks_file, header=0, sep='\t')
            for n in range(df.shape[0]):
                m = df.iloc[n, 1:].values.astype(bool)
                self.masks[df.iloc[n, 0]] = m
                cfg.input_length = len(m)
        else:
            if not hasattr(cfg, 'input_length'):
                raise ValueError('input_length needed when masks file not provided')

        if hasattr(cfg, 'moe') and cfg.moe:
            self.moe = True
            print(f'using Mixture of Experts')

        self.maps = {}
        self.expert_to_head = {}

        df = pd.read_table(maps_file, header=0, sep='\t')
        for n in range(df.shape[0]):
            head = df['head'].iloc[n]
            expert = df['expert'].iloc[n]
            label = df['label'].iloc[n]
            self.maps[head] = [label, expert]
            self.expert_to_head[expert] = head

        if cfg.backbone == 'mlp':
            backbone_class = MLPBackbone
        elif cfg.backbone == 'cnn':
            backbone_class = CNNBackbone
        elif cfg.backbone == 'spliceai':
            backbone_class = SpliceAIBackbone

        if not self.moe:
            if cfg.backbone == 'mlp':
                self.backbone = backbone_class(
                    input_channels=cfg.input_channels,
                    input_length=cfg.input_length,
                    hidden_dims=cfg.hidden_dims,
                    dropout=cfg.dropout,
                )
            elif cfg.backbone == 'cnn':
                self.backbone = backbone_class(
                    input_channels=cfg.input_channels,
                    input_length=cfg.input_length,
                    hidden_dims=cfg.hidden_dims,
                    kernel_sizes=cfg.kernel_sizes,
                    strides=cfg.strides,
                    dropout=cfg.dropout,
                    use_batchnorm=cfg.use_batchnorm,
                    global_pool=cfg.global_pool,
                )
            elif cfg.backbone == 'spliceai':
                self.backbone = backbone_class(cfg)
            self.heads = nn.ModuleDict({head: nn.Linear(self.backbone.output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})
        else:
            self.experts = nn.ModuleDict()
            for e in self.expert_to_head:
                mask = self.masks[e]
                if cfg.backbone == 'mlp':
                    expert = backbone_class(
                        input_channels=cfg.input_channels,
                        input_length=sum(mask),
                        hidden_dims=cfg.hidden_dims,
                        dropout=cfg.dropout,
                    )
                elif cfg.backbone == 'cnn':
                    expert = backbone_class(
                        input_channels=cfg.input_channels,
                        input_length=sum(mask),
                        hidden_dims=cfg.hidden_dims,
                        kernel_sizes=cfg.kernel_sizes,
                        strides=cfg.strides,
                        dropout=cfg.dropout,
                        use_batchnorm=cfg.use_batchnorm,
                        global_pool=cfg.global_pool,
                    )
                elif cfg.backbone == 'spliceai':
                    expert = backbone_class(cfg)
                self.experts[e] = expert
            self.heads = nn.ModuleDict({head: nn.Linear(self.experts[self.maps[head][-1]].output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the backbone(s) and every classification head.

        Args:
            x: Input tensor of shape `(batch, input_channels, input_length)`.
                When Mixture-of-Experts is enabled, each expert instead
                receives `x` masked along the length dimension by its
                boolean feature mask.

        Returns:
            Mapping of head name to a logits tensor of shape
            `(batch, num_classes, 2)`, where `num_classes = label + 1` and
            the last dimension indexes the two allele copies.
        """
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
    cfg.moe = True
    cfg.backbone_class = MLPBackbone

    model = HierarchicalHLA(cfg)
    data = torch.randn(4, 2, 1000)
    outputs = model(data)
    # print(outputs['HLA-A'].shape)
    # print(outputs['HLA-B'].shape)

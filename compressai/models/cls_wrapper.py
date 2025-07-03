import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import types
import math
import numpy as np
from typing import cast

from timm import create_model
from timm.models.layers import trunc_normal_, to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.tinylic_vr import TinyLICVR
from compressai.layers import ViTBlock, ResViTBlock, MultistageMaskedConv2d
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers

from .utils import conv, deconv, update_registered_buffers, quantize_ste, \
    Demultiplexer, Multiplexer, Demultiplexerv2, Multiplexerv2

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TransformModule(nn.Module):
    def __init__(self, in_ch, out_ch, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, upscale_factor=4):
        super().__init__()
        self.upsample_layer = conv(in_ch, out_ch * (upscale_factor ** 2), kernel_size=1, stride=1)
        self.transform_layer = ViTBlock(dim=out_ch,
                                        depth=depth,
                                        num_heads=num_heads,
                                        kernel_size=kernel_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate,
                                        norm_layer=norm_layer,)
        self.upscale_factor = upscale_factor

    def forward(self, x, quality):
        x = self.upsample_layer(x)
        x = F.pixel_shuffle(x, self.upscale_factor)
        x, _ = self.transform_layer(x, quality)
        return x


class ClsWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = TinyLICVR()
        self.adapter = TransformModule(320, 96, depth=1, num_heads=3)
        self.task_model = create_model('convnext_tiny', pretrained=True)
        self.task_model_teacher = create_model('convnext_tiny', pretrained=True).eval().requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 320))

        self.register_buffer('imnet_mean', torch.tensor(IMAGENET_DEFAULT_MEAN), persistent=False)
        self.register_buffer('imnet_std', torch.tensor(IMAGENET_DEFAULT_STD), persistent=False)

        torch.nn.init.normal_(self.mask_token, std=.02)

    @torch.no_grad()
    def forward_teacher(self, x):
        features = []
        x = self.task_model_teacher.stem(x)
        features.append(x.detach())
        for blk in self.task_model_teacher.stages:
            x = blk(x)
            features.append(x.detach())
        x = self.task_model_teacher.norm_pre(x)
        x = self.task_model_teacher.forward_head(x)
        return x, features

    def forward_task(self, x, quality):
        features = []
        x = self.adapter(x, quality)
        features.append(x)
        for blk in self.task_model.stages:
            x = blk(x)
            features.append(x)
        x = self.task_model.norm_pre(x)
        x = self.task_model.forward_head(x)
        return x, features

    def norm_img(self, x):
        return (x - self.imnet_mean.view(1, -1, 1, 1)) / self.imnet_std.view(1, -1, 1, 1)

    def unnorm_img(self, x):
        return x * self.imnet_std.view(1, -1, 1, 1) + self.imnet_mean.view(1, -1, 1, 1)

    def forward(self, x, quality=1, **kwargs):
        outs = self.backbone(x, quality, **kwargs)
        B, C, H, W = outs['latent']['y_hat'].shape
        y_hat = outs['latent']['y_hat']

        outs['pred'], features = self.forward_task(y_hat, quality)
        outs['x_hat'], outs['decisions']['dec'] = self.backbone.g_s(y_hat * torch.abs(self.backbone.InverseGain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3), quality)

        _, features_teacher = self.forward_teacher(self.norm_img(x))
        outs['dist_loss'] = sum(F.mse_loss(features[i], features_teacher[i]) for i in range(len(features)))

        return outs

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False, update_quantiles: bool = False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)#, update_quantiles=update_quantiles)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        with torch.cuda.amp.autocast(enabled=False):
            loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self):
        for param in self.task_model.parameters():
            param.requires_grad = False

    def unfreeze_task(self):
        for param in self.task_model.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

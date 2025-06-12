# CMD Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR 


# Helper modules for channel-wise pooling operations.
class ChannelMaxPooling3D(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.max_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        out = self.max_pooling(x)
        return out.permute(0, 4, 1, 2, 3)  # [B, C', D', H', W']

class ChannelAvgPooling3D(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg_pooling = nn.AvgPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        out = self.avg_pooling(x)
        return out.permute(0, 4, 1, 2, 3)


class T2FLAIR_3DFea(nn.Module):
    """
    3D Feature extractor for T2-FLAIR mismatch.
    
    This module computes features for the T2 and FLAIR MRI sequences using
    separate 3D convolutions, amplifies their difference by a factor γ,
    and then applies channel-wise max and average pooling to generate an
    attention map. This map is used to augment the original features.
    
    Parameters:
        in_ch (int): Number of input channels for each modality (default: 1).
        base_ch (int): Number of output channels for the initial convolution.
        diff_amp (float): Amplification factor (γ) for the difference.
    """
    def __init__(self, in_ch=1, base_ch=64, diff_amp=2.0):
        super().__init__()
        self.diff_amp = diff_amp
        self.conv1_t2 = nn.Conv3d(in_ch, base_ch, 7, 2, 3, bias=False)
        self.conv1_flair = nn.Conv3d(in_ch, base_ch, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(self.conv1_t2.weight)
        nn.init.kaiming_normal_(self.conv1_flair.weight)

        self.max_pool = ChannelMaxPooling3D((1, 1, 64), (1, 1, 64))
        self.avg_pool = ChannelAvgPooling3D((1, 1, 64), (1, 1, 64))

        self.conv2 = nn.Conv3d(2, 1, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t2_3d, flair_3d):
        feat_t2 = self.conv1_t2(t2_3d)
        feat_flair = self.conv1_flair(flair_3d)
        diff_feat = (feat_t2 - feat_flair) * self.diff_amp

        feat_max = self.max_pool(diff_feat)
        feat_avg = self.avg_pool(diff_feat)

        feat_cat = torch.cat((feat_max, feat_avg), dim=1)
        attn = self.relu2(self.conv2(feat_cat))
        attn_map = self.sigmoid(attn)

        feat_t2_aug = feat_t2 + attn_map * feat_t2
        feat_flair_aug = feat_flair + attn_map * feat_flair

        return feat_t2_aug, feat_flair_aug


class CMDModule(nn.Module):
    """
    Cross-Modality Differential (CMD) module with 4-way segmentation gating.
    """
    def __init__(
        self,
        backbone: nn.Module = None,
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=4,       
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=2,
        use_checkpoint=True,
        pretrained_path=None,
        diff_amp=2.0,
        min_gate=0.1,
        base_ch=64,
        **kwargs
    ):
        super().__init__()
        # instantiate 4-way segmentation backbone
        if backbone is None:
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,
                **kwargs
            )
        else:
            self.backbone = backbone

        self.min_gate = min_gate
        self.cmd_feature_extractor = T2FLAIR_3DFea(in_ch=1, base_ch=base_ch, diff_amp=diff_amp)

        # classification head (base_ch * 2 pooled features)
        self.classification_head = nn.Sequential(
            nn.Linear(base_ch * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.model_path = pretrained_path or ''

    def forward(self, x_in):
        """
        Args:
            x_in: [B, 4, D, H, W] (e.g. [FLAIR, T1, T1c, T2])
        Returns:
            seg_logits: [B, 4, D, H, W]
            cls_logits: [B, num_classes]
        """
        bsz = x_in.size(0)
        # --- Segmentation branch for gating ---
        hidden = self.backbone.swinViT(x_in, normalize=True)
        enc0 = self.backbone.encoder1(x_in)
        enc1 = self.backbone.encoder2(hidden[0])
        enc2 = self.backbone.encoder3(hidden[1])
        enc3 = self.backbone.encoder4(hidden[2])
        dec4 = self.backbone.encoder10(hidden[4])
        dec3 = self.backbone.decoder5(dec4, hidden[3])
        dec2 = self.backbone.decoder4(dec3, enc3)
        dec1 = self.backbone.decoder3(dec2, enc2)
        dec0 = self.backbone.decoder2(dec1, enc1)
        out = self.backbone.decoder1(dec0, enc0)
        seg_logits = self.backbone.out(out)  # [B, 4, D, H, W]

        # compute whole-tumor probability = sum of ET, ED, NCR/NET channels
        seg_prob = torch.softmax(seg_logits, dim=1)
        tumor_prob = seg_prob[:, 1:4, ...].sum(dim=1, keepdim=True)
        # soft-gate between min_gate and 1.0
        gate = self.min_gate + (1.0 - self.min_gate) * tumor_prob

        # extract FLAIR and T2, apply gating
        flair = x_in[:, 0:1, ...]
        t2 = x_in[:, 3:4, ...]
        gated_flair = flair * gate
        gated_t2 = t2 * gate

        # --- CMD branch ---
        feat_t2_aug, feat_flair_aug = self.cmd_feature_extractor(gated_t2, gated_flair)

        # pool and classify
        t2_pool = F.adaptive_avg_pool3d(feat_t2_aug, (1, 1, 1)).view(bsz, -1)
        flair_pool = F.adaptive_avg_pool3d(feat_flair_aug, (1, 1, 1)).view(bsz, -1)
        mismatch_feat = torch.cat([t2_pool, flair_pool], dim=1)
        cls_logits = self.classification_head(mismatch_feat)

        return seg_logits, cls_logits

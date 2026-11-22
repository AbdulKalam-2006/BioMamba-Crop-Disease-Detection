import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VascularCrossScanEngine(nn.Module):
    """
    Biological Vascular Cross-Scanning Engine:
    - 0° Horizontal Scan: Sweeps along lateral veins across the lamina.
    - 90° Vertical Scan: Sweeps along the primary central midrib (xylem & phloem).
    """
    def __init__(self):
        super(VascularCrossScanEngine, self).__init__()

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        scan_0 = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        scan_90 = x.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
        return torch.cat([scan_0, scan_90], dim=-1)

class SelectiveStateSpaceBlock(nn.Module):
    """
    Selective State-Space Recurrence (Mamba SSM) with linear O(N) complexity.
    """
    def __init__(self, d_model, d_state=16):
        super(SelectiveStateSpaceBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2)
        self.dt_proj = nn.Linear(d_state, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        x_and_res = self.in_proj(u)
        x, res = x_and_res.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x_act = F.silu(x_conv)

        ssm_params = self.x_proj(x_act)
        delta, _ = ssm_params.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(delta))

        decay = torch.sigmoid(dt)
        y = x_act * decay

        out = self.out_proj(y * F.silu(res))
        return self.norm(out)

class BioMamba(nn.Module):
    def __init__(self, num_classes=17, d_model=128, d_state=16):
        super(BioMamba, self).__init__()
        
        # 1. High-Precision Feature Extractor
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = base.features  # Outputs [B, 576, 7, 7]
        
        # Project backbone output to match vascular scanner
        self.feature_proj = nn.Sequential(
            nn.Conv2d(576, d_model // 2, kernel_size=1),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU()
        )

        # 2. Vascular Cross-Scanning Engine (0° & 90°)
        self.cross_scanner = VascularCrossScanEngine()
        self.proj_in = nn.Linear(d_model, d_model)

        # 3. Selective State-Space Core
        self.ssm_block = SelectiveStateSpaceBlock(d_model=d_model, d_state=d_state)

        # 4. Dense Classification Head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat_map = self.feature_proj(feat)           # [B, 64, 7, 7]
        tokens = self.cross_scanner(feat_map)        # [B, 49, 128]
        tokens = self.proj_in(tokens)
        
        ssm_out = self.ssm_block(tokens)             # [B, 49, 128]
        pooled = ssm_out.mean(dim=1)
        normed = self.norm(pooled)
        logits = self.classifier(self.dropout(normed))

        return logits, ssm_out, feat_map
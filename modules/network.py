import torch
import torch.nn as nn
from mamba_ssm import Mamba

class PatchEmbedding(nn.Module):

    def __init__(self,  in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))
        self.conv2_1 = nn.Sequential(nn.Conv2d(self.in_channels[1], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))
        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=False))

    def forward(self, x):
        x_h = self.conv1_2(self.conv1_1(x[0]))
        x_l = self.conv2_2(self.conv2_1(x[1]))

        x_out = torch.cat((x_h + x_l.mean(dim=1, keepdim=True).expand_as(x_h),
                                x_l + x_h.mean(dim=1, keepdim=True).expand_as(x_l)), dim=1)

        return self.conv3_1(x_out)

class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn_spatial_to_spectral = nn.MultiheadAttention(d_model, num_heads=4)
        self.cross_attn_spectral_to_spatial = nn.MultiheadAttention(d_model, num_heads=4)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_spectral = nn.LayerNorm(d_model)

    def forward(self, spatial_feat, spectral_feat):
        attn_spatial, _ = self.cross_attn_spatial_to_spectral(
            query=spatial_feat,
            key=spectral_feat,
            value=spectral_feat
        )
        spatial_out = self.norm_spatial(spatial_feat + attn_spatial)

        attn_spectral, _ = self.cross_attn_spectral_to_spatial(
            query=spectral_feat,
            key=spatial_feat,
            value=spatial_feat
        )
        spectral_out = self.norm_spectral(spectral_feat + attn_spectral)

        return spatial_out+spectral_out

class CrossModalMamba(nn.Module):
    """Mamba Fusion Module with specific dimension processing"""

    def __init__(self, size=9, out_channels=128):
        super().__init__()

        self.seq_length =size*size

        self.cross_fusion = CrossAttentionFusion(out_channels)

        self.mamba_forward = Mamba(
            d_model=out_channels,
            d_state=32,
            d_conv=4,
            expand=2
        )

        self.mamba_spectral = Mamba(
            d_model=self.seq_length,
            d_state=32,
            d_conv=4,
            expand=2
        )
        self.norm_pre = nn.LayerNorm(out_channels)
        self.norm_post = nn.LayerNorm(out_channels)


        self.pool = nn.AdaptiveAvgPool1d(1)


    def _reshape_to_sequence(self, x):
        return x.flatten(2).permute(0, 2, 1)

    def _reshape_to_spatial(self, x):
        return x.permute(0, 2, 1)

    def _spectral_scan(self, x):
        x_spectral=x.permute(0, 2, 1)
        spectral_out = self.mamba_spectral(x_spectral)

        return spectral_out.permute(0, 2, 1)



    def forward(self, x):

        seq1 = self._reshape_to_sequence(x)

        seq = self.norm_pre(seq1)

        mambaspa = self.mamba_forward(seq)

        mambaspe= self._spectral_scan(seq)

        mamba_out= self.cross_fusion(mambaspa, mambaspe)

        mamba_out = self.norm_post(mamba_out + seq)

        spatial_out = self._reshape_to_spatial(mamba_out)

        pooled = self.pool(spatial_out).squeeze(-1)

        return pooled

class Net(nn.Module):
    def __init__(self,  in_channels, n_class, dim_emebeding):
        super(Net, self).__init__()
        self.embedding_layer = PatchEmbedding( in_channels)

        self.mamba_path = CrossModalMamba()
        self.mamba_path = CrossModalMamba()
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)

    def forward(self, x_1, x_2):

        embedded_1 = self.embedding_layer(x_1)
        embedded_2 = self.embedding_layer(x_2)

        x_1 = self.mamba_path(embedded_1)
        x_2 = self.mamba_path(embedded_2)

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        return y_1, y_2

    def forward_embedding(self, x):
        h = self.mamba_path(self.embedding_layer(x))
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h = self.mamba_path(self.embedding_layer(x))
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels

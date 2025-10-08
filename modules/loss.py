import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)

        sim = torch.matmul(z, z.T) / self.temperature  # Dot similarity
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class ClusteringLoss(nn.Module):
    def __init__(self, weight_clu_loss=1, regularization_coef=0.05):
        super(ClusteringLoss, self).__init__()
        self.weight_clu_loss = weight_clu_loss
        self.regularization_coef = regularization_coef

    def forward(self, y_prob, cluster_center=None):
        batch_size, n_clusters = y_prob.shape

        target_prob = self._target_distribution(y_prob)

        kl_loss = torch.sum(target_prob * (torch.log(target_prob + 1e-10) - torch.log(y_prob + 1e-10)))
        kl_loss = self.weight_clu_loss * kl_loss / batch_size

        reg_loss = 0.
        if cluster_center is not None:
            U, S, V = torch.svd(cluster_center)
            reg_loss = torch.sum((S - 1.0) ** 2)

        cluster_probs = torch.mean(y_prob, dim=0)
        cluster_probs = cluster_probs / (torch.sum(cluster_probs) + 1e-10)

        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-10))
        entropy_reg = -(entropy_loss - math.log(n_clusters)) ** 2

        total_loss = kl_loss + self.regularization_coef * (reg_loss + entropy_reg)
        return total_loss

    def _target_distribution(self, q):

        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + 1e-10)
        return weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-10)

class JointLoss(nn.Module):
    """
    joint train model with a center-based loss plussed with a contrastive loss
    """
    def __init__(self, batch_size, lambda_=0.5, weight_clu=1, regularization_coef=0.05, device='cpu'):
        super(JointLoss, self).__init__()
        self.device = device
        self.weight_clu = weight_clu
        self.regularization_coef = regularization_coef
        self.criterion_contrastive = InstanceLoss(batch_size, lambda_, device).to(device)
        self.clustering_loss = ClusteringLoss(weight_clu, regularization_coef)

    def forward(self, y_1, y_2, cluster_center=None):
        h = torch.cat([y_1, y_2], dim=0)
        loss_con = self.criterion_contrastive(y_1, y_2)
        loss_clu = self.clustering_loss(h, cluster_center)
        loss = loss_con + loss_clu
        return loss, loss_con, loss_clu










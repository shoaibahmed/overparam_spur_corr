"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Imported from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='mean', normalize=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        assert reduction in ["mean", "none"]
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) == 2:
            # Aritifically increase the size of features to [bsz, 1, feature_size]
            features = features[:, None, :]
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        if self.normalize:
            # Normalize the features
            feature_norm = torch.linalg.vector_norm(features, ord=2, dim=2, keepdim=True)
            features = features / feature_norm

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            assert self.reduction == "none"
            loss = loss.mean(dim=0)
            assert len(loss.shape) == 1
            assert loss.shape == (len(features),)

        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, reduction='mean', use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        assert reduction in ["mean", "none"]
        self.reduction = reduction
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to("cuda" if self.use_gpu else "cpu"))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        assert x.shape[1] == self.feat_dim
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12)
        if self.reduction == 'mean':
            loss = dist.sum() / batch_size
        return loss


class CEWithCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambd, reduction='mean', use_gpu=True):
        super(CEWithCenterLoss, self).__init__()
        self.lambd = lambd
        self.num_classes = num_classes
        
        assert reduction in ["mean", "none"]
        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.center_criterion = CenterLoss(num_classes, feat_dim, reduction=reduction, use_gpu=use_gpu)

    def forward(self, x, y):
        assert isinstance(x, list) or isinstance(x, tuple)  # Features, logits
        features, logits = x
        ce_loss = self.ce_criterion(logits, y)
        center_loss = self.center_criterion(features, y)
        assert center_loss.shape == (len(features), self.num_classes), f"{center_loss.shape}"
        loss = ce_loss + self.lambd * center_loss.sum(dim=1)
        return loss


class DistillationLoss(nn.Module):
    def __init__(self, teacher_net, lambd, reduction='mean'):
        super(DistillationLoss, self).__init__()
        self.lambd = lambd
        
        assert reduction in ["mean", "none"]
        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.teacher_net = teacher_net
        self.teacher_net.requires_grad_(False)
        self.teacher_net.eval()
        self.reduction = reduction

    def forward(self, x, logits, y):
        with torch.no_grad():
            teacher_logits = self.teacher_net(x)
            assert not torch.isnan(teacher_logits).any()
            assert len(teacher_logits.shape) == 2
            teacher_preds = teacher_logits.argmax(dim=1)
            assert teacher_preds.shape == y.shape
            correct_teacher_preds = teacher_preds == y
            assert len(correct_teacher_preds.shape) == 1
            
            target_logits = logits.clone()
            target_logits[correct_teacher_preds] = teacher_logits[correct_teacher_preds]
            target_logits = target_logits.detach()
        
        ce_loss = self.ce_criterion(logits, y)
        if self.reduction == "mean":
            distillation_loss = self.mse_criterion(logits, target_logits)
        else:
            distillation_loss = ((logits - target_logits) ** 2).sum(dim=1)
            assert not torch.isnan(distillation_loss).any()
        loss = ce_loss + self.lambd * distillation_loss
        return loss


class DistillationWithCenterLoss(nn.Module):
    def __init__(self, teacher_net, num_classes, feat_dim, lambd_distill, lambd_center,
                 reduction='mean', use_gpu=True):
        super(DistillationWithCenterLoss, self).__init__()
        self.lambd_distill = lambd_distill
        self.lambd_center = lambd_center
        self.num_classes = num_classes
        
        assert reduction in ["mean", "none"]
        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.center_criterion = CenterLoss(num_classes, feat_dim, reduction=reduction, use_gpu=use_gpu)
        
        self.teacher_net = teacher_net
        self.teacher_net.requires_grad_(False)
        self.teacher_net.eval()
        self.reduction = reduction

    def forward(self, x, logits, y):
        assert isinstance(logits, list) or isinstance(logits, tuple)  # Features, logits
        features, logits = logits
        
        with torch.no_grad():
            teacher_logits = self.teacher_net(x)
            assert not torch.isnan(teacher_logits).any()
            assert len(teacher_logits.shape) == 2
            teacher_preds = teacher_logits.argmax(dim=1)
            assert teacher_preds.shape == y.shape
            correct_teacher_preds = teacher_preds == y
            assert len(correct_teacher_preds.shape) == 1
            
            target_logits = logits.clone()
            target_logits[correct_teacher_preds] = teacher_logits[correct_teacher_preds]
            target_logits = target_logits.detach()
        
        # CE loss
        ce_loss = self.ce_criterion(logits, y)
        
        # Center loss
        center_loss = self.center_criterion(features, y)
        assert center_loss.shape == (len(features), self.num_classes), f"{center_loss.shape}"
        
        # Distillation loss
        if self.reduction == "mean":
            distillation_loss = self.mse_criterion(logits, target_logits)
        else:
            distillation_loss = ((logits - target_logits) ** 2).sum(dim=1)
            assert not torch.isnan(distillation_loss).any()
        
        loss = ce_loss + self.lambd_center * center_loss.sum(dim=1) + self.lambd_distill * distillation_loss
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction):
        super(BCELoss, self).__init__()
        assert reduction in ['mean', 'none']
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, x, y):
        assert x.shape == y.shape, f"{x.shape} != {y.shape}"
        out = -(y * torch.log(x + self.eps) + (1 - y) * torch.log(1 - x + self.eps))
        if self.reduction == 'mean':
            return out.mean()
        assert self.reduction == 'none'
        return out.mean(dim=1)  # Reduce only on the number of classes, but not on the instances

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True

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

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

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
            mask = mask.float().to(device).clone()

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

        # assert batch_size == mask.shape[0], print('number of samples in batch size does not match contrastive matrix for supervised contrastive loss.')

        if self.first_run:
            print('in loss : feature shape ', features.shape)
            print('in loss : contrastive matrix shape ', mask.shape)
            
            
        # compute logits
        
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
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
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()

        self.first_run = False

        return loss
    
class MySupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True
        self.sim_function = nn.CosineSimilarity(2)
        
    def forward(self,features, positive_mask, negative_mask, neutral_mask = None):
        
        ## features shape extended_batch, d_model
        ## mask shape extended_batch,extended_batch
        
        ## add zeros to negative and positive masks to prevent self-contrasting
        
        self_contrast = (~(torch.eye(positive_mask.shape[0], device = features.device).bool())).int()
        
        
        positive_mask = positive_mask * self_contrast
        negative_mask = negative_mask * self_contrast
        
    
        original_cosim = self.sim_function(features.unsqueeze(1),features.unsqueeze(0))/self.temperature
        logits_max, _ = torch.max(original_cosim, dim=1, keepdim=True)
        original_cosim = original_cosim - logits_max.detach()
        
        
        
        original_cosim = torch.exp(original_cosim)
        positives = original_cosim * positive_mask
        negatives = original_cosim * negative_mask
        
        
        negatives_summed = negatives.sum(1, keepdim = True)
        
        log_negatives_summed = torch.log(negatives_summed)
        
        
        log_prob = positives - log_negatives_summed
    
        positive_cardinal = positive_mask.sum(1)
        positive_cardinal[positive_cardinal == 0] = 1
        
        
        
        # print(positive_cardinal)
        # print(positives_negged.sum(1))
        
        log_prob_sum = (log_prob * positive_mask).sum(1)
        
        
        
        loss = - log_prob_sum/positive_cardinal
        
        if loss.isnan().any():
            zero_index = (positive_cardinal).nonzero().flatten()
            plt.imsave('wtf.png',positive_mask.cpu().numpy())

        loss = loss.mean()
        return loss
    
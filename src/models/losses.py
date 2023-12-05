import torch
import torch.nn as nn
import matplotlib.pyplot as plt

    
class MySupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True
        self.sim_function = nn.CosineSimilarity(2)
        
    def get_similarities(self, features, temperature = None):
        if temperature is None:
            temperature = self.temperature  
        return self.sim_function(features.unsqueeze(1),features.unsqueeze(0))/temperature
        
    def forward(self,features, positive_mask, negative_mask, neutral_mask = None):
        
        ## features shape extended_batch, d_model
        ## mask shape extended_batch,extended_batch
        
        ## add zeros to negative and positive masks to prevent self-contrasting
        
        self_contrast = (~(torch.eye(positive_mask.shape[0], device = features.device).bool())).int()
        
        
        positive_mask = positive_mask * self_contrast
        negative_mask = negative_mask * self_contrast
        
    
        original_cosim = self.get_similarities(features=features)
        # logits_max, _ = torch.max(original_cosim, dim=1, keepdim=True)
        # original_cosim = original_cosim - logits_max.detach()
        
        
        
        positives = original_cosim * positive_mask
        negatives = torch.exp(original_cosim) * negative_mask
        
        
        negatives_summed = negatives.sum(1, keepdim = True)
        
        log_negatives_summed = torch.log(negatives_summed)
        
        
        log_prob = positives - log_negatives_summed
        
        positive_cardinal = positive_mask.sum(1)
        
        log_prob = positives - log_negatives_summed
    
        # positive_cardinal = positive_mask.sum(1)
        # positive_cardinal[positive_cardinal == 0] = 1
        
        
        
        # print(positive_cardinal)
        # print(positives_negged.sum(1))
        
        # log_prob_sum = (log_prob * positive_mask).sum(1)
        
        # print an error warning if positive_cardinal has a zero element
        
        
        # loss = - log_prob_sum/positive_cardinal
        
        loss = - log_prob.mean()
        return loss
    
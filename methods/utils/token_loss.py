from timm import loss
from timm.data.transforms_factory import transforms_imagenet_train
import torch
from torch.functional import Tensor
import torch.nn as nn

class AdaLoss(nn.Module):
    def __init__(self, base_criterion, 
                token_target_ratio=0.5, 
                token_loss_ratio=2., 
                token_minimal=0., 
                token_minimal_weight=0.
                ):
        
        super().__init__()
        self.base_criterion = base_criterion
        
        self.token_target_ratio = token_target_ratio
        self.token_loss_ratio = token_loss_ratio
        self.token_minimal = token_minimal
        self.token_minimal_weight = token_minimal_weight


    def forward(self, outputs, y):
        '''
        head_select: (b, num_layers, num_head)
        '''

        x, token_select, _ = outputs["prediction"], outputs["token_select"], outputs["token_logits"]

        base_loss = self.base_criterion(x, y)
        # layer_loss = self._get_layer_loss(x, layer_select, layer_logits)
        token_loss = self._get_token_loss(x, token_select)
        
        loss = base_loss +  self.token_loss_ratio * token_loss

        return loss, dict(base_loss=base_loss, token_loss=self.token_loss_ratio * token_loss)
    
    def _get_token_loss(self, x, token_select):
        """
        token_select : tensor (b, num_layer, l)

        """
        if token_select is not None :
            token_mean = token_select.mean()
            # token_flops_loss = (token_mean - self.token_target_ratio).abs().mean()
            # token_flops_loss = (token_mean - self.token_target_ratio).clamp(min=0.).mean()
            token_flops_loss = ((token_mean - self.token_target_ratio)**2).mean()

            if self.token_minimal_weight > 0 :
                token_mean = token_select.mean(-1)
                token_minimal_loss = (self.token_minimal - token_mean).clamp(min=0.).sum()
            else :
                token_minimal_loss = 0

            token_loss = token_flops_loss + self.token_minimal_weight * token_minimal_loss
        else :
            token_loss = x.new_zeros(1).mean()

        return token_loss
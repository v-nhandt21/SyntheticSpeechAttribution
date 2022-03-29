import torch

class FocalLoss(torch.nn.Module):
     '''Multi-class Focal loss implementation'''
     def __init__(self, gamma=2):
          super(FocalLoss, self).__init__()
          self.gamma = gamma

     def forward(self, input, target):
          """
          input: [N, C]
          target: [N, ]
          """
          logpt = torch.nn.functional.log_softmax(input, dim=1)
          pt = torch.exp(logpt)
          logpt = (1-pt)**self.gamma * logpt
          loss = torch.nn.functional.nll_loss(logpt, target)
          return loss

class LabelSmoothingLoss(torch.nn.Module):
     def __init__(self, smoothing: float = 0.1, 
               reduction="mean", weight=None):
          super(LabelSmoothingLoss, self).__init__()
          self.smoothing   = smoothing
          self.reduction = reduction
          self.weight    = weight

     def reduce_loss(self, loss):
          return loss.mean() if self.reduction == 'mean' else loss.sum() \
          if self.reduction == 'sum' else loss

     def linear_combination(self, x, y):
          return self.smoothing * x + (1 - self.smoothing) * y

     def forward(self, preds, target):
          assert 0 <= self.smoothing < 1

          if self.weight is not None:
               self.weight = self.weight.to(preds.device)

          n = preds.size(-1)
          log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
          loss = self.reduce_loss(-log_preds.sum(dim=-1))
          nll = torch.nn.functional.nll_loss(
               log_preds, target, reduction=self.reduction, weight=self.weight
          )
          return self.linear_combination(loss / n, nll)
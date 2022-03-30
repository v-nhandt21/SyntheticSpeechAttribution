import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
     r"""Implement of large margin arc distance: :
          Args:
               in_features: size of each input sample
               out_features: size of each output sample
               s: norm of input feature
               m: margin
               cos(theta + m)
          """
     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
          super(ArcMarginProduct, self).__init__()
          self.in_features = in_features
          self.out_features = out_features
          self.s = s
          self.m = m
          self.weight = Parameter(torch.FloatTensor(out_features, in_features))
          nn.init.xavier_uniform_(self.weight)

          self.easy_margin = easy_margin
          self.cos_m = math.cos(m)
          self.sin_m = math.sin(m)
          self.th = math.cos(math.pi - m)
          self.mm = math.sin(math.pi - m) * m

     def forward(self, input, label):
          # --------------------------- cos(theta) & phi(theta) ---------------------------
          cosine = F.linear(F.normalize(input), F.normalize(self.weight))
          sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
          phi = cosine * self.cos_m - sine * self.sin_m
          if self.easy_margin:
               phi = torch.where(cosine > 0, phi, cosine)
          else:
               phi = torch.where(cosine > self.th, phi, cosine - self.mm)
          # --------------------------- convert label to one-hot ---------------------------
          # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
          one_hot = torch.zeros(cosine.size(), device='cuda')
          one_hot.scatter_(1, label.view(-1, 1).long(), 1)
          # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
          output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
          output *= self.s
          # print(output)

          return output
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
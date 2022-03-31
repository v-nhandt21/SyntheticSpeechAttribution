from efficientnet_pytorch import EfficientNet
import torch
from efficientNetv2 import effnetv2_m

class Classifier(torch.nn.Module):
     def __init__(self):
          super(Classifier, self).__init__()
          model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)

          #self.conv2d = torch.nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          self.backbone = model

     def forward(self, x): 
          #x = self.conv2d(x.unsqueeze(1))
          x = x.unsqueeze(1).repeat(1,3,1,1) 
          x = self.backbone(x)
          return x

class Cascade(torch.nn.Module):
     def __init__(self, unseen, h):
          super(Cascade, self).__init__()
          if unseen:
               self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
          else:
               if h.backbone == "efficientNetv2":
                    self.model =  effnetv2_m(num_classes=5)
               else:
                    self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=5)
     def forward(self, x): 
          x = x.unsqueeze(1).repeat(1,3,1,1) 
          x = self.model(x)
          return x

class Tulet(torch.nn.Module):
     def __init__(self):
          super(Tulet, self).__init__()
          self.model = EfficientNet.from_pretrained('efficientnet-b2')
     def forward(self, x): 
          x = x.unsqueeze(1).repeat(1,3,1,1) 
          x = self.model(x)
          return x
          
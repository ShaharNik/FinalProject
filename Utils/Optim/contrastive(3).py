import torch
import torch.nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

  """
  Contrastive loss function.
  Based on:
  """
  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
  

  # def forward(self, x1, x2, y):
  #   # euclidian distance
  #   diff = x2 - x1
  #   dist_sq = torch.sum(torch.pow(diff, 2), 1)
  #   dist = torch.sqrt(dist_sq)

  #   loss = 0.5 * ((y * dist) + ((1-y)* torch.pow(F.relu(self.margin-dist),2)))
  #   return loss.mean()


  def forward(self, output1, output2, target):
    ContrastiveLoss = self.Contrastive(output1, output2, target)

    # distances = (output2 - output1).pow(2).sum(1)  # squared distances
    # losses = 0.5 * (target.float() * distances + (1.0 - target.float()).float() * F.relu(self.margin - (distances).sqrt()).pow(2))
    # return losses.mean()
      
  
  def Contrastive(self, output1, output2, target):
    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances + (1.0 - target.float()).float() * F.relu(self.margin - (distances).sqrt()).pow(2))
    return losses.mean()


  # def arcFace(self,output1, output2, target):
    




    # def __init__(self, margin=1.0):
    #     super(ContrastiveLoss, self).__init__()
    #     self.margin = margin

    # def check_type_forward(self, in_types):
    #     assert len(in_types) == 3

    #     x0_type, x1_type, y_type = in_types
    #     assert x0_type.size() == x1_type.shape
    #     assert x1_type.size()[0] == y_type.shape[0]
    #     assert x1_type.size()[0] > 0
    #     assert x0_type.dim() == 2
    #     assert x1_type.dim() == 2
    #     assert y_type.dim() == 1

    # def forward(self, x0, x1, y):
    #     self.check_type_forward((x0, x1, y))

    #     # euclidian distance
    #     diff = x0 - x1
    #     dist_sq = torch.sum(torch.pow(diff, 2), 1)
    #     dist = torch.sqrt(dist_sq)

    #     mdist = self.margin - dist
    #     dist = torch.clamp(mdist, min=0.0)
    #     loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    #     loss = torch.sum(loss) / 2.0 / x0.size()[0]
    #     return loss
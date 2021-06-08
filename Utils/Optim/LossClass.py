import torch
import torch.nn
import torch.nn.functional as F

class LossClass(torch.nn.Module):

  def __init__(self, contrastiveMargin=1.0, arcMargin=0.5,arcScale=30):
    super(LossClass, self).__init__()
    self.contrastiveMargin = contrastiveMargin
    self.arcMargin = arcMargin
    self.arcScale = arcScale

  def forward(self, output1, output2, target):
    ContrastiveLoss = self.Contrastive(output1, output2, target)
    ArcFaceLoss = self.ArcFace(output1, output2, target)
    TotalLoss = ContrastiveLoss + 0.5*(ArcFaceLoss)
    # print("Contrastive loss => {}".format(ContrastiveLoss))
    # print("ArcFaceLoss loss => {}".format(ArcFaceLoss))
    # print("TotalLoss loss => {}".format(TotalLoss))
    return TotalLoss
      
  
  def Contrastive(self, output1, output2, target):
    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances + (1.0 - target.float()).float() * F.relu(self.contrastiveMargin - (distances).sqrt()).pow(2))
    return losses.mean()


  def ArcFace(self, output1, output2, label ):
    label = label.view(label.shape[1])
    output1 = F.normalize(output1,p=2,dim=-1)
    index = torch.where(label != -1)[0]
    m_hot = torch.zeros(index.size()[0], output1.size()[1], device=output1.device)
    # print(label.shape)
    # print(index.shape)
    # print(m_hot.shape)
    m_hot.scatter_(1, label[index, None].to(torch.int64), self.arcMargin)
    output1.acos_()
    output1[index] += m_hot
    output1.cos_().mul_(self.arcScale)

    output2 = F.normalize(output2,p=2,dim=-1)
    index = torch.where(label != -1)[0]
    m_hot = torch.zeros(index.size()[0], output2.size()[1], device=output2.device)
    m_hot.scatter_(1, label[index, None].to(torch.int64), self.arcMargin)
    output2.acos_()
    output2[index] += m_hot
    output2.cos_().mul_(self.arcScale)
    # print("Output 1 => {} and output 2 => {}".format(output1,output2))
    
    meanO1O2 = (output1+output2).mean()
    # print("Output 1 and output 2 mean => {}".format(meanO1O2))
    # print(meanO1O2)


    return meanO1O2

    

# class ArcFaceTwo(nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFaceTwo, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, cosine: torch.Tensor, label):
#         # cosine.cos_()
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine.acos_()
#         cosine[index] += m_hot
#         cosine.cos_().mul_(self.s)
#         return cosine




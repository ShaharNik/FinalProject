import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PairSet(Dataset):
  def __init__(self,path,transform,mod):
    self.dir_path = path
    self.transform = transform
    self.imgsLabls={}
    self.pairs={}
    self.labels=None
    self.mod = mod
    self.load()


  def __getitem__(self,idx):
    return self.pairs[idx]


  def __len__(self):
    return len(self.pairs)


  def load(self):
    self.labels = os.listdir(self.dir_path)
    # print(self.labels)
    index=0
    for label in self.labels:
      path = os.path.join(self.dir_path, label)
      images = os.listdir(path)
      for image_id in images:
        img_path = os.path.join(path, image_id)
        self.imgsLabls[index] = (self.transform(Image.open(img_path)),label)
        index+=1
    # print(index)
    # print(len(self.imgsLabls))
    index =0
    for i in range(len(self.imgsLabls)):
      for j in range(len(self.imgsLabls)):
        if i!=j:
          xi,yi = self.imgsLabls[i]
          xj,yj = self.imgsLabls[j]
          if(not (yi==yj) and (index % self.mod == 0)):
            continue
          self.pairs[index] = (xi,xj,(yi==yj))
          index+=1
    print(index)      
        
  






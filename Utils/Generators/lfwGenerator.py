import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class lfwGenerator(Dataset):
  def __init__(self,pathToData,pathToPairs,transform):
    self.pathToData = pathToData
    self.transform = transform
    self.pathToPairs = pathToPairs
    self.imgsLabls={}
    self.pairs={}
    self.labels=None
    self.makeParisNew()
    # self.loadAllData()
    # self.makePairs()
  
  
  def __getitem__(self,idx):
    return self.pairs[idx]


  def __len__(self):
    return len(self.pairs)
  

  def loadAllData(self):
    self.labels = os.listdir(self.pathToData)
    indexPic = 0
    for label in self.labels:
      print(label, indexPic)
      indexPic+=1
      path = os.path.join(self.pathToData, label)
      images = os.listdir(path)
      tempDict = {}
      index=1
      for image_id in images:
        img_path = os.path.join(path, image_id)
        tempDict[index] = self.transform(Image.open(img_path))
        index+=1
      
      self.imgsLabls[label] = tempDict.copy()
      #print(tempDict)

  def makePairs(self):

    index = 0
    fileas = open(self.pathToPairs, 'r')
    f1 = fileas.readlines()
    for x in f1:
      #print(x)
      splited = x.split()
      if (len(splited) == 3):
        tempdict = self.imgsLabls[splited[0]]
        #print(tempdict.shape)
        #print(splited[1])
        #print(tempdict[int(splited[1])])
        p1 = tempdict[int(splited[1])]
        p2 = tempdict[int(splited[2])]
        self.pairs[index] = (p1, p2, splited[0], splited[0])
      else:
        tempdictp1 = self.imgsLabls[splited[0]]
        tempdictp2 = self.imgsLabls[splited[2]]
        p1 = tempdictp1[int(splited[1])]
        p2 = tempdictp2[int(splited[3])]
        self.pairs[index] = (p1, p2, splited[0], splited[2])
      index += 1


    #self.labels = os.listdir(self.pathToData)

  def makeParisNew(self):
    # self.labels = os.listdir(self.pathToData)
    fileas = open(self.pathToPairs, 'r')
    f1 = fileas.readlines()
    index = 0
    for x in f1:
      if index%100==0:
        print(index)
      splited = x.split()
      if (len(splited) == 3):
        path = os.path.join(self.pathToData, splited[0])
        images = os.listdir(path)
        img1_path = os.path.join(path,images[int(splited[1])-1])
        img1 = self.transform(Image.open(img1_path))
        img2_path = os.path.join(path,images[int(splited[2])-1])
        img2 = self.transform(Image.open(img2_path))
        self.pairs[index] = (img1, img2, splited[0], splited[0])
      else:
        path1 = os.path.join(self.pathToData, splited[0])
        path2 = os.path.join(self.pathToData, splited[2])
        images1 = os.listdir(path1)
        images2 = os.listdir(path2)
        img1_path = os.path.join(path1,images1[int(splited[1])-1])
        img1 = self.transform(Image.open(img1_path))
        img2_path = os.path.join(path2,images2[int(splited[3])-1])
        img2 = self.transform(Image.open(img2_path))
        self.pairs[index] = (img1, img2, splited[0], splited[2])
      index+=1








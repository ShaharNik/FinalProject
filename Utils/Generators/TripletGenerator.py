import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from itertools import permutations, islice
from torch.utils.data import Dataset
import random
from random import sample


class TripletImageLoader():
  def __init__(self,path,transform,TrainRatio=0.8):
    self.path = path
    self.transform = transform
    #self.triplets = []
    self.trainTriplets = []
    self.testTriplets = []
    self.TrainRatio = TrainRatio
    self.generateTriplets()
    # self.generateTripletsFlickr()

  
  def generateTriplets(self):
    triplets = []
    labels = os.listdir(self.path)
    indexPic = 0
    OptinalLabels = []
    picWithOne = []
    for label in labels:
      pathToLabel = os.path.join(self.path, label)
      images = os.listdir(pathToLabel)
      if len(images)>=2:
        OptinalLabels.append(pathToLabel)
      else:
        pathnew = os.path.join(pathToLabel, images[0])
        # print(pathnew)
        picWithOne.append(pathnew)
    
    indexTriplets = 0
    indexToFinish =0 
    for path in OptinalLabels:
      # if indexToFinish == 185: ######## TO DELETE! - 185 is able to run
      #   break
      images = os.listdir(path)
      imgPermute = list(permutations(images,2))
      random.shuffle(imgPermute)
      imgPermute = imgPermute[0:20]
      for i, (anchor,pos) in enumerate(imgPermute):
        negativeSample = random.sample(picWithOne,5)
        anchorPicPath = os.path.join(path, anchor)
        posPicPath = os.path.join(path, pos)
        #anchorPic = self.transform(Image.open(anchorPicPath))
        #posPic = self.transform(Image.open(posPicPath))
        for sample in negativeSample:
          negPicPath = os.path.join(path, sample)
          #negPic = self.transform(Image.open(negPicPath))
          triplets.append((anchorPicPath,posPicPath,negPicPath))
          indexTriplets+=1
      indexToFinish+=1
      print("{} is done triplets so far - {}".format(indexToFinish,indexTriplets))
    random.shuffle(triplets)
    length = len(triplets)
    self.trainTriplets = triplets[0:int((length*self.TrainRatio))]
    self.testTriplets =  triplets[int((length*self.TrainRatio)):]
    del triplets
    print(indexTriplets)

  def generateTripletsFlickr(self):
    triplets = []
    labels = os.listdir(self.path)
    indexPic = 0
    OptinalLabels = []
    picWithOne = []
    for label in labels:
      pathToLabel = os.path.join(self.path, label)
      OptinalLabels.append(pathToLabel)

    
    indexTriplets = 0
    indexToFinish =0 
    for path in OptinalLabels:
      # if indexToFinish == 185: ######## TO DELETE! - 185 is able to run
      #   break
      images = os.listdir(path)
      imgPermute = list(permutations(images,2))
      random.shuffle(imgPermute)
      imgPermute = imgPermute[0:20]
      for i, (anchor,pos) in enumerate(imgPermute):
        negativeSample = random.sample(OptinalLabels,5)
        anchorPicPath = os.path.join(path, anchor)
        posPicPath = os.path.join(path, pos)
        #anchorPic = self.transform(Image.open(anchorPicPath))
        #posPic = self.transform(Image.open(posPicPath))
        for sample in negativeSample:
          if sample != path:
            jjjj = os.listdir(sample)
            jjjj = random.choice(jjjj)
            negPicPath = os.path.join(sample, jjjj)
            #negPic = self.transform(Image.open(negPicPath))
            triplets.append((anchorPicPath,posPicPath,negPicPath))
            # print(anchorPicPath,posPicPath,negPicPath)
            indexTriplets+=1
      indexToFinish+=1
      print("{} is done triplets so far - {}".format(indexToFinish,indexTriplets))
    random.shuffle(triplets)
    length = len(triplets)
    self.trainTriplets = triplets[0:int((length*self.TrainRatio))]
    self.testTriplets =  triplets[int((length*self.TrainRatio)):]
    del triplets
    print(indexTriplets)

    

class TripletImage(Dataset):
  def __init__(self,data,transform,train=True):
    self.transform = transform
    if train:
      self.data = data.trainTriplets 
    else:
      self.data = data.testTriplets 
    
  def __getitem__(self,idx):
    #return self.data[idx]
    anchorPicPath,posPicPath,negPicPath = self.data[idx]
    anchorPic = self.transform(Image.open(anchorPicPath))
    posPic = self.transform(Image.open(posPicPath))
    negPic = self.transform(Image.open(negPicPath))
    return (anchorPic,posPic,negPic)


  def __len__(self):
    return len(self.data)




    



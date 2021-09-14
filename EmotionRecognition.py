import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Lambda
import time
import pandas as pd
import copy
from PIL import Image
from skimage import io
from MLP import MLP
from skimage.feature import hog
from skimage import data, exposure
from skimage import io, img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
import cv2
from torch.utils.data import TensorDataset
from skimage.measure import label, regionprops
from matplotlib import rc
import matplotlib.animation as animation
from matplotlib import patches

import os
import pickle

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/Computer Vision/Coursework/' 
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))

class CustomImageDataset():
    def __init__(self, annotations_df, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        #image = Image.open(img_path)
        #image = torch.from_numpy(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)


        sample = {"image": image, "label": label}
        return image, label

class CustomHOGSDataset(Dataset):

    def __init__(self, annotations_df, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        h = image.copy()
        #image = Image.open(img_path)
        #image = torch.from_numpy(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        inp = image.numpy().transpose((1, 2, 0))
        HOG_des = hog(inp, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1),block_norm = "L1", multichannel=True)
      
        sample = {"image": image, "label": label, "HOG_des": HOG_des}
        return image, label, HOG_des


        
class EmotionRecognition():


    ### TAKEN FROM LAB 9 ###
    def imshow(inp, title=None):
     """Imshow for Tensor."""
     inp = inp.numpy().transpose((1, 2, 0))
     inp = np.asarray(data_stds) * inp + np.asarray(data_means)
     inp = np.clip(inp, 0, 1)       # Clip to intensity outliers to [0, 1] range
     plt.imshow(inp)
     if title is not None:
        plt.title(title)
        
    def EmotionRecognition(path_to_dataset, model_type):
    
        device = 'cpu'
        model_type = model_type
        data_dir = path_to_dataset
        l_test = os.path.join(data_dir, "labels", "list_label_test.txt")
        d_test = os.path.join(data_dir, "test")
        
        test_lab = pd.read_csv(l_test, delimiter= ' ',header=None)
        test_lab = test_lab.rename(columns={0: "name", 1: "lbl"})
        test_lab['name'] = test_lab['name'].str[:-4] + "_aligned" + ".jpg"
        test_lab['lbl'] = test_lab['lbl'] - 1
        
        data_means = [0.485, 0.456, 0.406]
        data_stds = [0.229, 0.224, 0.225]
        if model_type == "HOGS": 
            test_data = CustomHOGSDataset(test_lab, d_test, transform=transforms.Compose([transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(data_means, data_stds)]))
        else:
            
            test_data = CustomImageDataset(test_lab, d_test, transform=transforms.Compose([transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(data_means, data_stds)]))
        if model_type == "SIFT": 
            ds_tst = DataLoader(test_data, batch_size =4, shuffle=False)
            def sift_conv(ds, b_num):
              des_list = []
              l_set = []
              i_set = []
              for image, label in ds: #taking images and applying sift 
                for i in range(b_num-1):
                    
                  #print(image[i].shape
                  inp = image[i].numpy().transpose((1, 2, 0))
                  image8bit = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                  sift = cv2.SIFT_create()
                  SIFT_kp, SIFT_des = sift.detectAndCompute(image8bit, None)
                  if SIFT_des is not None:
                      des_list.append(SIFT_des)
                      l_set.append(label[i])
              des_array = np.vstack(des_list)
              return des_array, des_list, l_set
            test_arr, test_list, test_labels = sift_conv(ds_tst, 4)
            def kmeans_bovw(des_array, des_list):
              batch_size = des_array.shape[0] // 4
              kmeans = MiniBatchKMeans(n_clusters=70, batch_size=batch_size).fit(des_array) #Doing KMeans for BoVW
            
              # Convert descriptors into histograms of codewords for each image
              hist_list = []
              idx_list = []
            
              for des in des_list:
                  hist = np.zeros(70)
            
                  idx = kmeans.predict(des)
                  idx_list.append(idx)
                  for j in idx:
                      hist[j] = hist[j] + (1 / len(des))
                  hist_list.append(hist)
            
              hist_array = np.vstack(hist_list)
              return hist_array

            test_bovw = kmeans_bovw(test_arr, test_list)
            tst_ds = TensorDataset(torch.tensor(test_bovw), torch.tensor(test_labels))

            ds_test = DataLoader(tst_ds, batch_size=4, shuffle=True)
        else:
            ds_test = DataLoader(test_data, batch_size =1, shuffle=True)
        test_dataset_sizes = len(test_data.img_labels)
        
        if model_type == "CNN": 
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features

            model.fc = nn.Linear(num_ftrs,7)
            model = model.to(device)
            mdl = torch.load(str(os.path.join(GOOGLE_DRIVE_PATH,'fin_CNN.pth')), map_location='cpu')
            
        if model_type == "HOGS":
            model = MLP(input_dim = 1152)
            mdl = torch.load(str(os.path.join(GOOGLE_DRIVE_PATH,'fin_HOG_MLP.pth')), map_location='cpu')
            
        if model_type == "SIFT":
            model = MLP(input_dim = 70)
            mdl = torch.load(str(os.path.join(GOOGLE_DRIVE_PATH,'fin_SIFT_MLP.pth')), map_location='cpu')
            
        if model_type == "HOGS":
            criterion = nn.CrossEntropyLoss()
            running_loss = 0
            running_corrects = 0
            for inputs, labels, desc in ds_test:
                    # move data to GPU
                    inputs = desc.to(device)
                    labels = labels.to(device)
    
                    # forward
                    # track history if only in train
                    outputs = mdl(inputs.float())
                   
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    # statistics
                    running_loss += loss.item() * desc.size(0)
                    running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / test_dataset_sizes
            epoch_acc = running_corrects.double() / test_dataset_sizes
            print(f"test loss: {epoch_loss:.4f} test acc: {epoch_acc:.4f}")
        elif model_type == "CNN":
            criterion = nn.CrossEntropyLoss()
            running_loss = 0
            running_corrects = 0
            for inputs, labels in ds_test:
                    # move data to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # forward
                    # track history if only in train
                    outputs = mdl(inputs.float())
                   
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / test_dataset_sizes
            epoch_acc = running_corrects.double() / test_dataset_sizes
            print(f"test loss: {epoch_loss:.4f} test acc: {epoch_acc:.4f}")
            
        elif model_type == "SIFT":
            criterion = nn.CrossEntropyLoss()
            running_loss = 0
            running_corrects = 0
            for inputs, labels in ds_test:
                    # move data to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # forward
                    # track history if only in train
                    outputs = mdl(inputs.float())
                   
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / test_dataset_sizes
            epoch_acc = running_corrects.double() / test_dataset_sizes
            print(f"test loss: {epoch_loss:.4f} test acc: {epoch_acc:.4f}")
            
        ### TAKEN FROM LAB 9 ###
        images_so_far = 0
        num_images=4
        if model_type == "HOGS": 
            for inputs, labels, desc in ds_test:
                    inputs = inputs.to(device)
                    desc = desc.to(device)
                    labels = labels.to(device)
    
                    outputs = mdl(desc.float())
                    _, preds = torch.max(outputs, 1)
    
                    for j in range(inputs.size()[0]):
                        if images_so_far> num_images-1:
                          break
                        images_so_far += 1
                        ax = plt.subplot(num_images//2, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title(f"Predicted: {preds[j]}, Truth: {labels[j]}")
                        inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                        inp = np.asarray(data_stds) * inp + np.asarray(data_means)
                        inp = np.clip(inp, 0, 1)       # Clip to intensity outliers to [0, 1] range
                        plt.imshow(inp)
        elif model_type == "CNN": 
            for inputs, labels in ds_test:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    outputs = mdl(inputs)
                    _, preds = torch.max(outputs, 1)
    
                    for j in range(inputs.size()[0]):
                        if images_so_far> num_images-1:
                          break
                        images_so_far += 1
                        ax = plt.subplot(num_images//2, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title(f"Predicted: {preds[j]}, Truth: {labels[j]}")
                        inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                        inp = np.asarray(data_stds) * inp + np.asarray(data_means)
                        inp = np.clip(inp, 0, 1)       # Clip to intensity outliers to [0, 1] range
                        plt.imshow(inp)


        
class EmotionRecognitionVideo():
    
    def EmotionRecognitionVideo(video_name): 
        device = 'cpu'
        
        data_means = [0.485, 0.456, 0.406]
        data_stds = [0.229, 0.224, 0.225]
        ### ADAPTED FROM LAB 7 ###
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs,7)
        model = model.to(device)
        mdl = torch.load(str(os.path.join(GOOGLE_DRIVE_PATH,'fin_CNN.pth')), map_location='cpu')
        
        cap = cv2.VideoCapture(os.path.join(GOOGLE_DRIVE_PATH, video_name))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        in_transforms = transforms.Compose([
                transforms.Normalize(data_means, data_stds)])
        fc = 0
        ret = True
        mapping = {0:"surprise", 1:"fear", 2: "disgust", 3:"happiness", 4:"sadness", 5:"anger", 6:"neutral"}
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        while fc < frameCount and ret:
            ret, video[fc] = cap.read()
            video[fc] = cv2.cvtColor(video[fc], cv2.COLOR_BGR2RGB)
           
            # Pass frame to the classifier
            faces = face_cascade.detectMultiScale(video[fc], 1.2, 1)
            
           # https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
           
            # Extract bounding boxes for any faces identified
            for (x,y,w,h) in faces:
                cv2.rectangle(video[fc], (x, y), (x+w, y+h), (0, 0, 0), 2)
                f_data = video[fc][y:y+h, x:x+h].copy()
        
                #print(f_data.shape)
                b = cv2.dnn.blobFromImage(f_data, 1, swapRB=False, crop=False)
                
        
              #  print(b)
                b = torch.tensor(b)
                b = in_transforms(b)
              #  print(b)
                b = b.to(device)
                outputs = mdl(b)
                _, preds = torch.max(outputs, 1)
                p = preds.item()
                #print(p)
                cv2.putText(video[fc], mapping[p], (x,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0 ),1)
            fc += 1
        
            
        
        cap.release()

        rc('animation', html='jshtml')
        
        fig, ax = plt.subplots(figsize=(20,10))
        
        def frame(i):
        
            ax.clear()
            ax.axis('off')
            fig.tight_layout()
            plot=ax.imshow(video[i, :, :, :])
            return plot
        
        anim = animation.FuncAnimation(fig, frame, frames=180)
        plt.show(anim)
        return anim
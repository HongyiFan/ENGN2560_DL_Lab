import pickle
import numpy as np

import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from PIL import Image

#
# Helper functions to read a subset of CIFAR-10 dataset
#

def s(x): return(np.shape(x))

def load_cifar10_data(filename):
    
    with open('/content/data/cifar-10-batches-py/'+ filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data']
    labels = batch['labels']
    return features, labels

def return_photo(batch_file):

    assert batch_file.shape[1] == 3072
    
    dim = np.sqrt(1024).astype(int)
    r = batch_file[:, 0:1024].reshape(batch_file.shape[0], dim, dim, 1)
    g = batch_file[:, 1024:2048].reshape(batch_file.shape[0], dim, dim, 1) 
    b = batch_file[:, 2048:3072].reshape(batch_file.shape[0], dim, dim, 1)
    photo = np.concatenate([r,g,b], -1)
    
    return photo

# Load files
batch_1, labels_1 = load_cifar10_data('data_batch_1')
batch_2, labels_2 = load_cifar10_data('data_batch_2')
batch_3, labels_3 = load_cifar10_data('data_batch_3')
batch_4, labels_4 = load_cifar10_data('data_batch_4')
batch_5, labels_5 = load_cifar10_data('data_batch_5')

test, label_test = load_cifar10_data('test_batch')

X_train = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)
Y_train = np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0)

X_train = return_photo(X_train)

X_test = return_photo(test)
np.shape(X_train),np.shape(Y_train) 

filter=(Y_train==0)+(Y_train==1)+(Y_train==2)

cifar3_labels=Y_train[filter]

cifar3_imgs=X_train[filter]

label_test=np.array(label_test)

filter=(label_test==0)+(label_test==1)+(label_test==2)

test_cifar3_labels=label_test[filter]
test_cifar3_imgs=X_test[filter]

s(cifar3_imgs), s(cifar3_labels), s(test_cifar3_imgs), s(test_cifar3_labels)

class CIFAR10_from_array(Dataset):

    def __init__(self, data, label, transform=None):
        
        self.data = data
        self.label = label
        self.transform = transform
        self.img_shape = data.shape
        
    def __getitem__(self, index):
        
        img = Image.fromarray(self.data[index])
        label = self.label[index]
        
        if self.transform is not None:
          img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
            #label = torch.from_numpy(label).long()
        
        return img, label
        
    def __len__(self):
        
        return len(self.data)
    
    def plot_image(self, number):

        file = self.data
        label = self.label
        fig = plt.figure(figsize = (3,2))
        plt.imshow(file[number])
        plt.title(classes[label[number]])

def get_train_loader(size, batch_size):
  
  transform = transforms.Compose([
    transforms.RandomResizedCrop(size), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
  trainset = CIFAR10_from_array(data=cifar3_imgs, label=cifar3_labels, transform=transform)   

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
  
  return(trainloader)

def get_test_loader(size, batch_size):
  
  transform = transforms.Compose([
    transforms.RandomResizedCrop(size), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
  testset = CIFAR10_from_array(data=test_cifar3_imgs, label=test_cifar3_labels, transform=transform)
  
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

  return(testloader)

#!/usr/bin/env python
# coding: utf-8

# In[50]:


import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib import ticker
from scipy.signal import savgol_filter

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[25]:


### READ DATA ###
data = pd.read_csv('adult.csv')
print(data.info())


# In[26]:


print(data.shape)
print(data.columns)
print(data[:5])
income = pd.Index([">50K", "<=50K"])
data.income.value_counts()


# In[27]:


### DATA CLEANING AND CHECK ###
col_names = data.columns
num_rows = data.shape[0]

for feature in col_names: 
    count = data[feature].isin(["?"]).sum() #sum the "?"s in the column  
#     print("?s in", feature, ":", count)
    

# Get rid of samples with"?"  
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data = data[data[feature] != "?"]
    
data.income.value_counts()


# In[28]:


for feature in col_names: 
    count = data[feature].isin(["?"]).sum() #sum the "?"s in the column  
#     print("?s in", feature, ":", count)
    
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

#Check each possible value of a categorical feature and how often it occurs
for feature in categorical_feats:
    print("-----",feature,"-----")
    print(data[feature].value_counts()) # No.of each distinct feature/feature column
    
    n = len(pd.unique(data[feature]))
 


# In[29]:


### ENCODE DATA ###
# 1. Encode categorical features, drop the income
label_encoder = LabelEncoder() 
cts_feats = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
cat_data = data.drop(columns = cts_feats)
cat_data_encoded = cat_data.apply(label_encoder.fit_transform)
labels = cat_data_encoded['income'].values.reshape(45222, 1)  # labels in columns
cat_data_encoded = cat_data_encoded.drop(columns = 'income')


one_hot_encoder = OneHotEncoder()
cat_one_hot = one_hot_encoder.fit(cat_data_encoded.values)
cat_one_hot = one_hot_encoder.transform(cat_data_encoded.values).toarray() #98 features


# 2. Normalize continuous features
cts_data = data.drop(columns=categorical_feats)
cts_mean = cts_data.mean(axis = 0)
cts_std = cts_data.std(axis = 0)
cts_data = cts_data.sub(cts_mean,axis=1)
cts_data = cts_data.div(cts_std,axis=1)

X = np.concatenate([cts_data, cat_one_hot, labels], axis = 1) 
# X = np.concatenate([cts_data, cat_one_hot], axis = 1) 
X_high = X[X[:,104] == 1] #11208
X_low = X[X[:,104] == 0]
print(X_high.shape)
print(X_low.shape)

X_male = X[X[:,62] == 1]
X_female = X[X[:,62] == 0]
print(X_male.shape)


# In[30]:


# Define Adult Dataset class 
class AdultDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # generates one sample of data
        return torch.from_numpy(self.X[index]),self.y[index]


# In[31]:


# Set the protion of different labels, Encode the training data into a required class
def load_data(portion: int):
    #portion of the rich
    n = 5500
    n_high = int(n/100*portion)
    n_low = n - n_high
    sample_train = None
    
    
    np.random.shuffle(X_high)
    np.random.shuffle(X_low)
    sample_train = np.concatenate((X_high[:n_high], X_low[:n_low]), 0)
    
    train_X = sample_train[:,:104]
    train_y = sample_train[:,104:105].reshape(n)
    train_data = AdultDataset(train_X, train_y)
    

    
#     np.random.shuffle(X)
#     sample_test = X[1000:1200] 
#     test_X = sample_test[:,:104]
#     test_y = sample_test[:,104:105].reshape(200)
#     test_data = AdultDataset(test_X, test_y) 
       
#     train_loader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
#     test_loader = DataLoader(dataset=test_data,batch_size=32,shuffle=True)
    return train_data


# In[32]:


# Define the architecture of the network 
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(104, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
#         return torch.sigmoid(self.fc4(x))
        return self.fc4(x)
        


# In[33]:


#Define the accuracy on the test set
def accuracy(model, val_loader):
    model.eval()
    total_corr = 0
    for i,data in enumerate(val_loader):
        inputs,labels = data
        y_pred = model(inputs.float())
        for i in range (len(labels)):
            #<=50k is encoded to 0 and > 50k encoded to 1
            if (y_pred[i].item() > 0.5):
                r = 1
            else:
                r = 0
            if (r == labels[i].item()):
                total_corr += 1

    return float(total_corr)/len(val_loader.dataset)


# In[35]:


x_high = X_high[:500,:104]
y_high = X_high[:500,104:105]
high = AdultDataset(x_high, y_high)
high_loader = DataLoader(dataset = high, batch_size=32, shuffle=True)
    
x_low = X_low[:500,:104]
y_low = X_low[:500,104:105]
low = AdultDataset(x_low, y_low)
low_loader = DataLoader(dataset = low, batch_size=32, shuffle=True)


# In[65]:


# Initialize a network and specify the loss function and optimizer|binary cross entrophy and Adam optimizer
def get_model_paras(portion: int):
    print('Portion: '+ str(portion))
    lr = 0.001 #learning rate 
    epochs = 30

    train_dataset, test_dataset = torch.utils.data.random_split(load_data(portion = portion), [5000, 500])
    train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=True)
#     train_loader, test_loader = load_data(portion = portion)
    
    model = net()
    #CrossEntropyLoss takes batch_size*class_number tensor as pred, batch_size 1D tensor as target
    criterion = nn.BCEWithLogitsLoss()
#     criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-2)
    model.train()
    loss_list = []

    for epoch in range(epochs):
            running_loss = 0       
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data.float()).squeeze()
                loss = criterion(output.float(), target.float())
                loss.backward()
                optimizer.step()            
                running_loss += loss.item()
            
            loss_list.append(running_loss)
    plt.plot(loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Crossentropy Loss')
    plt.show()
    
    layers = []
    i = 0
    for name, param in model.named_parameters():
        print(name)
        print(param.size())
        if(i%2 == 0):
            layers.append(param.data)
        else:
            k = int((i-1)/2)
            layers[k] = torch.cat((layers[k], param.data.unsqueeze(1)), 1) 
        i += 1
    
#     acc = accuracy(model, test_loader)
# #     print('Accuracy on test set: {}'.format(acc))  
    
#     acc_high = accuracy(model, high_loader)
# #     print('Accuracy on high test set: {}'.format(acc_high))  
#     acc_low = accuracy(model, low_loader)
# #     print('Accuracy on low test set: {}'.format(acc_low)) 
# #     print('------------------------------------------')
    return acc, acc_high, acc_low


# In[66]:


a,b,c = get_model_paras(portion = 10)


# In[46]:


acc_l = []
high_acc = []
low_acc = []
for i in range(100):
    acc, acc_high, acc_low = (get_model_paras(portion = i))
    acc_l.append(acc)
    high_acc.append(acc_high)
    low_acc.append(acc_low)


# In[60]:


x = list(range(100))
for i,k in enumerate(x):
    x[i]=k/100
plt.plot(x,acc_l, label = 'Test set with the same data distribution')
plt.plot(x,high_acc, label = 'Test set with high income')
plt.plot(x,low_acc, label = 'Test set with low income')
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.legend(loc = 'best')
plt.xlabel('Portion')
plt.ylabel('Accuracy on different test sets')
plt.savefig('Accuracy on different test sets', dpi=500, bbox_inches='tight')
plt.show()


# In[65]:


for i in range (2000):
    torch.save(get_model_paras(portion = i%100), 'meta_train/'+ str(i) +'.pt')    
    ## e.g. if the index is 218, the percentage should be 18%


# In[ ]:





import numpy as np
import os
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt

train_x = pickle.load(open('./non_redundant_train_mfcc_features_0.pickle', 'rb'))
test_x = pickle.load(open('./non_redundant_test_mfcc_features_0.pickle', 'rb'))
train_y = pickle.load(open('./non_redundant_training_labels_flattened_2d.pickle', 'rb'))
test_y = pickle.load(open('./non_redundant_test_labels_flattened_2d.pickle', 'rb'))
img_train_y_ = pickle.load(open('non_redundant_training_labels_flattened_3d.pickle', 'rb'))
img_test_y_ = pickle.load(open('non_redundant_test_labels_flattened_3d.pickle', 'rb'))

img_train = pickle.load(open('./non_redundant_training_features_flattened_3d.pickle', 'rb'))
img_test = pickle.load(open('./non_redundant_test_features_flattened_3d.pickle', 'rb'))

print(img_train[0].shape)

audio_train_x = []
audio_train_y = []
audio_test_x = []
audio_test_y = []

for i in range(len(train_x)):
    feat = train_x[i].flatten()
    if(feat.shape[0]==3887):
        audio_train_x.append(feat)
        audio_train_y.append(train_y[i]-1)
    else:
    	audio_train_x.append(np.zeros(3887))

for i in range(len(test_x)):
    feat = test_x[i].flatten()
    audio_test_x.append(feat)
    audio_test_y.append(test_y[i]-1)

audio_train_x = np.asarray(audio_train_x).reshape((len(audio_train_x), 3887))
audio_test_x = np.asarray(audio_test_x).reshape((len(audio_test_x), 3887))

audio_train_x = preprocessing.scale(audio_train_x)
audio_test_x = preprocessing.scale(audio_test_x)

print(audio_train_x.shape, audio_test_x.shape)

img_train_x = np.zeros((len(img_train), 4*2048))
img_train_y = []

for i, feat in enumerate(img_train):
    if(feat.shape[0]<img_train_x.shape[1]):
        feat =  np.concatenate((feat, np.zeros(int(img_train_x.shape[1])-int(feat.shape[0]))), axis=0)
        img_train_x[i] = feat
        img_train_y.append(img_train_y_[i]-1)
    else:
        feat = feat[:int(img_train_x.shape[1])]
        img_train_x[i] = feat
        img_train_y.append(img_train_y_[i]-1)

img_test_x = np.zeros((len(img_test), 4*2048))
img_test_y = []

for i, feat in enumerate(img_test):
    if(feat.shape[0]<img_test_x.shape[1]):
        feat =  np.concatenate((feat, np.zeros(int(img_test_x.shape[1])-int(feat.shape[0]))), axis=0)
        img_test_x[i] = feat
        img_test_y.append(img_test_y_[i]-1)
    else:
        feat = feat[:int(img_test_x.shape[1])]
        img_test_x[i] = feat
        img_test_y.append(img_test_y_[i]-1)

img_train_x = preprocessing.scale(img_train_x)
img_test_x = preprocessing.scale(img_test_x)

print(img_train_x.shape, img_test_x.shape)

train_x = audio_train_x
test_x = audio_test_x
# train_x = img_train_x
# test_x = img_test_x


svm = SVC(gamma='auto')
svm.fit(train_x, labs)

print(accuracy_score(labs, svm.predict(train_x)))
print(accuracy_score(test_labs, svm.predict(test_x)))

log_reg = LogisticRegression()
log_reg.fit(train_x, labs)

print(accuracy_score(labs, log_reg.predict(train_x)))
print(accuracy_score(test_labs, log_reg.predict(test_x)))

dt = tree.DecisionTreeClassifier()
dt.fit(train_x, labs)

print(accuracy_score(labs, dt.predict(train_x)))
print(accuracy_score(test_labs, dt.predict(test_x)))

rf = RandomForestClassifier()
rf.fit(train_x, labs)

print(accuracy_score(labs, rf.predict(train_x)))
print(accuracy_score(test_labs, rf.predict(test_x)))


############################ AUDIO MLP MODEL ####################################

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

class Classification_Net(nn.Module):
    def __init__(self, input_dims=3887):
        super(Classification_Net, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 8)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, feats):
        concat_vector = feats
        out1 = self.dropout_5(self.relu(self.fc1(concat_vector)))
        out2 = self.dropout_2(self.relu(self.fc2(out1)))
        out3 = self.dropout_2(self.relu(self.fc3(out2)))
        out4 = self.fc4(out3)
        return out4

from torch.utils.data import *

train_tensor_x = torch.Tensor(audio_train_x)
train_tensor_y = torch.Tensor(labs)
test_tensor_x = torch.Tensor(audio_test_x)
test_tensor_y = torch.Tensor(test_labs)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

model = Classification_Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# model.cuda()

losses = []
val_accs = []
train_accs = []

for epoch in range(50):
      for itr, (x, y) in enumerate(train_dataloader):
        # x = x.cuda()
        # y = y.cuda()

        outputs = model(x)
        loss = criterion(outputs, y.long())

        # params = list(model.parameters())
        # l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

        # for param in params[1:]:
        #   l1_regularization += torch.norm(param, 1)
        #   l2_regularization += torch.norm(param, 2)

        # reg_1 = Variable(l1_regularization)
        # reg_2 = Variable(l2_regularization)

        # loss += reg_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if(itr%100 == 0):
          with torch.no_grad():
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0

            for x, y in test_dataloader:
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x)

              _, predicted = torch.max(outputs.data, 1)
              val_total += y.size(0)
              val_correct += (predicted == y.long()).sum().item()

            for i, (x, y) in enumerate(train_dataloader):
              if(i == 50):
                break
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x)

              _, predicted = torch.max(outputs.data, 1)
              train_total += y.size(0)
              train_correct += (predicted == y.long()).sum().item()
              
            print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Val acc: '+str(100 * val_correct/val_total)+', Train acc: '+str(100 * train_correct/train_total))
            losses.append(loss.item())
            val_accs.append(100 * val_correct/val_total)
            train_accs.append(100 * train_correct/train_total)
            scheduler.step(100 * val_correct/val_total)
torch.save(model.state_dict(), 'model_concat_16384_dimensional.ckpt')

plt.figure()
plt.plot(losses)
plt.title('Losses vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('unimodal_audio_loss.png')

plt.figure()
plt.plot(val_accs, label='Test')
plt.plot(train_accs, label='Train')
plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.legend(loc='best')
plt.savefig('unimodal_audio_acc.png')

# # ############################ IMAGE MLP MODEL ####################################

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

class Classification_Net(nn.Module):
    def __init__(self, input_dims=8192):
        super(Classification_Net, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 8)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, feats):
        concat_vector = feats
        out1 = self.dropout_5(self.relu(self.fc1(concat_vector)))
        out2 = self.dropout_2(self.relu(self.fc2(out1)))
        out3 = self.dropout_2(self.relu(self.fc3(out2)))
        out4 = self.fc4(out3)
        return out4

from torch.utils.data import *

train_tensor_x = torch.Tensor(img_train_x)
train_tensor_y = torch.Tensor(img_train_y)
test_tensor_x = torch.Tensor(img_test_x)
test_tensor_y = torch.Tensor(img_test_y)

print(train_tensor_x.shape, train_tensor_y.shape, test_tensor_x.shape, test_tensor_y.shape)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

model = Classification_Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# model.cuda()

losses = []
val_accs = []
train_accs = []

for epoch in range(50):
      for itr, (x, y) in enumerate(train_dataloader):
        # x = x.cuda()
        # y = y.cuda()

        outputs = model(x)
        loss = criterion(outputs, y.long())

        # params = list(model.parameters())
        # l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

        # for param in params[1:]:
        #   l1_regularization += torch.norm(param, 1)
        #   l2_regularization += torch.norm(param, 2)

        # reg_1 = Variable(l1_regularization)
        # reg_2 = Variable(l2_regularization)

        # loss += reg_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if(itr%100 == 0):
          with torch.no_grad():
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0

            for x, y in test_dataloader:
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x)

              _, predicted = torch.max(outputs.data, 1)
              val_total += y.size(0)
              val_correct += (predicted == y.long()).sum().item()

            for i, (x, y) in enumerate(train_dataloader):
              if(i == 50):
                break
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x)

              _, predicted = torch.max(outputs.data, 1)
              train_total += y.size(0)
              train_correct += (predicted == y.long()).sum().item()
              
            print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Val acc: '+str(100 * val_correct/val_total)+', Train acc: '+str(100 * train_correct/train_total))
            losses.append(loss.item())
            val_accs.append(100 * val_correct/val_total)
            train_accs.append(100 * train_correct/train_total)
            # scheduler.step(100 * val_correct/val_total)
# torch.save(model.state_dict(), 'model_concat_16384_dimensional.ckpt')

plt.figure()
plt.plot(losses)
plt.title('Losses vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('unimodal_img_loss.png')

plt.figure()
plt.plot(val_accs, label='Test')
plt.plot(train_accs, label='Train')
plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.legend(loc='best')
plt.savefig('unimodal_img_acc.png')

################################ EARLY FUSION MODEL ####################################

class TextAndImageDataset(Dataset):
	def __init__(self, data, labels):
		self.img_data = data[0]
		self.text_data = data[1]
		self.img_labels = labels[0]
		self.text_labels = labels[1]

	def __len__(self):
		return len(self.img_data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		img = self.img_data[idx]
		img_l = self.img_labels[idx]
		text = self.text_data[idx]
		text_l = self.text_labels[idx]

		return ([img, text], [img_l, text_l])

class Concat_Net(nn.Module):
    def __init__(self, img_input_dims=8192, audio_input_dims=3887):
        super(Concat_Net, self).__init__()
        self.fc1 = nn.Linear(audio_input_dims+img_input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4_audio = nn.Linear(64, 8)
        self.fc4_img = nn.Linear(64, 8)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, img_feats, audio_feats):
        concat_vector = torch.cat((img_feats, audio_feats), dim=1)
        out1 = self.dropout_5(self.relu(self.fc1(concat_vector.float())))
        out2 = self.dropout_2(self.relu(self.fc2(out1)))
        out3 = self.dropout_2(self.relu(self.fc3(out2)))
        out_img = self.fc4_img(out3)
        out_audio = self.fc4_audio(out3)
        return out_img, out_audio

from torch.utils.data import *

train_data = [img_train_x, audio_train_x]
train_labels = [img_train_y, img_train_y]
test_data = [img_test_x, audio_test_x]
test_labels = [img_test_y, img_test_y]

train_dataset = TextAndImageDataset(train_data, train_labels)
test_dataset = TextAndImageDataset(test_data, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)

model = Concat_Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# model.cuda()

losses = []
img_train_acc = []
img_test_acc = []
audio_train_acc = []
audio_test_acc = []

for epoch in range(50):
      for itr, (x, y) in enumerate(train_dataloader):
        # x = x.cuda()
        # y = y.cuda()

        outputs_img, outputs_audio = model(x[0], x[1])
        loss = criterion(outputs_audio, y[1].long()) + criterion(outputs_img, y[0].long())

        # params = list(model.parameters())
        # l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

        # for param in params[1:]:
        #   l1_regularization += torch.norm(param, 1)
        #   l2_regularization += torch.norm(param, 2)

        # reg_1 = Variable(l1_regularization)
        # reg_2 = Variable(l2_regularization)

        # loss += reg_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if(itr%100 == 0):
          with torch.no_grad():
            train_correct_img = 0
            train_total_img = 0
            val_correct_img = 0
            val_total_img = 0
            train_correct_audio = 0
            train_total_audio = 0
            val_correct_audio = 0
            val_total_audio = 0
            
            for x, y in test_dataloader:
              # x = x.cuda()
              # y = y.cuda()

              outputs_img, outputs_audio = model(x[0], x[1])

              _, predicted_img = torch.max(outputs_img.data, 1)
              val_total_img += y[0].size(0)
              val_correct_img += (predicted_img == y[0].long()).sum().item()

              _, predicted_audio = torch.max(outputs_audio.data, 1)
              val_total_audio += y[1].size(0)
              val_correct_audio += (predicted_audio == y[1].long()).sum().item()

            for i, (x, y) in enumerate(train_dataloader):
              if(i == 50):
                break
              # x = x.cuda()
              # y = y.cuda()

              outputs_img, outputs_audio = model(x[0], x[1])

              _, predicted_img = torch.max(outputs_img.data, 1)
              train_total_img += y[0].size(0)
              train_correct_img += (predicted_img == y[0].long()).sum().item()

              _, predicted_audio = torch.max(outputs_audio.data, 1)
              train_total_audio += y[1].size(0)
              train_correct_audio += (predicted_audio == y[1].long()).sum().item()

            print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Val img acc: '+str(100 * val_correct_img/val_total_img)+', Train img acc: '+str(100 * train_correct_img/train_total_img)+', Val audio acc: '+str(100 * val_correct_audio/val_total_audio)+', Train audio acc: '+str(100 * train_correct_audio/train_total_audio))
            losses.append(loss.item())
            audio_test_acc.append(100*val_correct_audio/val_total_audio)
            audio_train_acc.append(100*train_correct_audio/train_total_audio)
            img_test_acc.append(100*val_correct_img/val_total_img)
            img_train_acc.append(100*train_correct_img/train_total_img)   
            # scheduler.step(100 * val_correct/val_total)
# torch.save(model.state_dict(), 'model_concat_16384_dimensional.ckpt')

plt.figure()
plt.plot(losses)
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('losses.png')

plt.figure()
plt.plot(audio_test_acc)
plt.title('Audio Val accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio val accuracy')
plt.savefig('audio_test_acc.png')

plt.figure()
plt.plot(audio_train_acc)
plt.title('Audio Train accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio train accuracy')
plt.savefig('audio_train_acc.png')

plt.figure()
plt.plot(img_test_acc)
plt.title('Image Val accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image val accuracy')
plt.savefig('image_test_acc.png')

plt.figure()
plt.plot(img_test_acc)
plt.title('Image Train accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image train accuracy')
plt.savefig('image_train_acc.png')

plt.figure()
plt.plot(img_train_acc, label='Image')
plt.plot(audio_train_acc, label='Audio')
plt.title('Train accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Train accuracy')
plt.legend(loc='best')
plt.savefig('train_acc.png')

plt.figure()
plt.plot(img_test_acc, label='Image')
plt.plot(audio_test_acc, label='Audio')
plt.title('Test accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
plt.savefig('test_acc.png')

plt.figure()
plt.plot(img_train_acc, label='Train')
plt.plot(img_test_acc, label='Test')
plt.title('Image accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image accuracy')
plt.legend(loc='best')
plt.savefig('image_acc.png')

plt.figure()
plt.plot(audio_train_acc, label='Train')
plt.plot(audio_test_acc, label='Test')
plt.title('Audio accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio accuracy')
plt.legend(loc='best')
plt.savefig('audio_acc.png')

plt.figure()
plt.plot(audio_train_acc, label='Audio Train')
plt.plot(audio_test_acc, label='Audio Test')
plt.plot(img_train_acc, label='Image Train')
plt.plot(img_test_acc, label='Image Test')
plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('all_acc.png')

# pickle.dump(img_test_y, open('./img_test_y.pickle', 'wb'))
# pickle.dump(img_train_y, open('./img_train_y.pickle', 'wb'))
# pickle.dump(labs, open('./audio_train_y.pickle', 'wb'))
# pickle.dump(test_labs, open('./audio_test_y.pickle', 'wb'))


################################ HYBRID FUSION MODEL #########################################

class Concat_Net_2(nn.Module):
    def __init__(self, img_input_dims=8192, audio_input_dims=3887):
        super(Concat_Net_2, self).__init__()
        self.fc1 = nn.Linear(audio_input_dims, 512)
        self.fc2 = nn.Linear(img_input_dims, 512)
        self.fc3 = nn.Linear(512*2, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5_audio = nn.Linear(64, 8)
        self.fc5_img = nn.Linear(64, 8)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, img_feats, audio_feats):
        out_img1 = self.dropout_5(self.relu(self.fc2(img_feats.float())))
        out_audio1 = self.dropout_5(self.relu(self.fc1(audio_feats.float())))
        concat_vector = torch.cat((out_img1, out_audio1), dim=1)
        out2 = self.dropout_2(self.relu(self.fc3(concat_vector)))
        out3 = self.dropout_2(self.relu(self.fc4(out2)))
        out_img = self.fc5_img(out3)
        out_audio = self.fc5_audio(out3)
        return out_img, out_audio

from torch.utils.data import *

train_data = [img_train_x, audio_train_x]
train_labels = [img_train_y, img_train_y]
test_data = [img_test_x, audio_test_x]
test_labels = [img_test_y, img_test_y]

train_dataset = TextAndImageDataset(train_data, train_labels)
test_dataset = TextAndImageDataset(test_data, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)

model = Concat_Net_2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# model.cuda()

losses = []
img_train_acc = []
img_test_acc = []
audio_train_acc = []
audio_test_acc = []

for epoch in range(50):
      for itr, (x, y) in enumerate(train_dataloader):
        # x = x.cuda()
        # y = y.cuda()

        outputs_img, outputs_audio = model(x[0], x[1])
        loss = criterion(outputs_audio, y[1].long()) + criterion(outputs_img, y[0].long())

        # params = list(model.parameters())
        # l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

        # for param in params[1:]:
        #   l1_regularization += torch.norm(param, 1)
        #   l2_regularization += torch.norm(param, 2)

        # reg_1 = Variable(l1_regularization)
        # reg_2 = Variable(l2_regularization)

        # loss += reg_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if(itr%100 == 0):
          with torch.no_grad():
            train_correct_img = 0
            train_total_img = 0
            val_correct_img = 0
            val_total_img = 0
            train_correct_audio = 0
            train_total_audio = 0
            val_correct_audio = 0
            val_total_audio = 0
            
            for x, y in test_dataloader:
              # x = x.cuda()
              # y = y.cuda()

              outputs_img, outputs_audio = model(x[0], x[1])

              _, predicted_img = torch.max(outputs_img.data, 1)
              val_total_img += y[0].size(0)
              val_correct_img += (predicted_img == y[0].long()).sum().item()

              _, predicted_audio = torch.max(outputs_audio.data, 1)
              val_total_audio += y[1].size(0)
              val_correct_audio += (predicted_audio == y[1].long()).sum().item()

            for i, (x, y) in enumerate(train_dataloader):
              if(i == 50):
                break
              # x = x.cuda()
              # y = y.cuda()

              outputs_img, outputs_audio = model(x[0], x[1])

              _, predicted_img = torch.max(outputs_img.data, 1)
              train_total_img += y[0].size(0)
              train_correct_img += (predicted_img == y[0].long()).sum().item()

              _, predicted_audio = torch.max(outputs_audio.data, 1)
              train_total_audio += y[1].size(0)
              train_correct_audio += (predicted_audio == y[1].long()).sum().item()

            print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Val img acc: '+str(100 * val_correct_img/val_total_img)+', Train img acc: '+str(100 * train_correct_img/train_total_img)+', Val audio acc: '+str(100 * val_correct_audio/val_total_audio)+', Train audio acc: '+str(100 * train_correct_audio/train_total_audio))
            losses.append(loss.item())
            audio_test_acc.append(100*val_correct_audio/val_total_audio)
            audio_train_acc.append(100*train_correct_audio/train_total_audio)
            img_test_acc.append(100*val_correct_img/val_total_img)
            img_train_acc.append(100*train_correct_img/train_total_img)   
            # scheduler.step(100 * val_correct/val_total)
# torch.save(model.state_dict(), 'model_concat_16384_dimensional.ckpt')

plt.figure()
plt.plot(losses)
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('losses_2.png')

plt.figure()
plt.plot(audio_test_acc)
plt.title('Audio Val accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio val accuracy')
plt.savefig('audio_test_acc_2.png')

plt.figure()
plt.plot(audio_train_acc)
plt.title('Audio Train accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio train accuracy')
plt.savefig('audio_train_acc_2.png')

plt.figure()
plt.plot(img_test_acc)
plt.title('Image Val accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image val accuracy')
plt.savefig('image_test_acc_2.png')

plt.figure()
plt.plot(img_test_acc)
plt.title('Image Train accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image train accuracy')
plt.savefig('image_train_acc_2.png')

plt.figure()
plt.plot(img_train_acc, label='Image')
plt.plot(audio_train_acc, label='Audio')
plt.title('Train accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Train accuracy')
plt.legend(loc='best')
plt.savefig('train_acc_2.png')

plt.figure()
plt.plot(img_test_acc, label='Image')
plt.plot(audio_test_acc, label='Audio')
plt.title('Test accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
plt.savefig('test_acc_2.png')

plt.figure()
plt.plot(img_train_acc, label='Train')
plt.plot(img_test_acc, label='Test')
plt.title('Image accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Image accuracy')
plt.legend(loc='best')
plt.savefig('image_acc_2.png')

plt.figure()
plt.plot(audio_train_acc, label='Train')
plt.plot(audio_test_acc, label='Test')
plt.title('Audio accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Audio accuracy')
plt.legend(loc='best')
plt.savefig('audio_acc_2.png')

plt.figure()
plt.plot(audio_train_acc, label='Audio Train')
plt.plot(audio_test_acc, label='Audio Test')
plt.plot(img_train_acc, label='Image Train')
plt.plot(img_test_acc, label='Image Test')
plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('all_acc_2.png')


############################### MTL MODEL ###############################

class Concat_Net_3(nn.Module):
    def __init__(self, img_input_dims=8192, audio_input_dims=3887):
        super(Concat_Net_3, self).__init__()
        self.fc1 = nn.Linear(audio_input_dims, 512)
        self.fc2 = nn.Linear(img_input_dims, 512)
        self.fc3 = nn.Linear(512*2, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5_audio = nn.Linear(64, 32)
        self.fc5_img = nn.Linear(64, 32)
        self.merge_fc = nn.Linear(64, 8)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, img_feats, audio_feats):
        out_img1 = self.dropout_5(self.relu(self.fc2(img_feats.float())))
        out_audio1 = self.dropout_5(self.relu(self.fc1(audio_feats.float())))
        concat_vector = torch.cat((out_img1, out_audio1), dim=1)
        out2 = self.dropout_2(self.relu(self.fc3(concat_vector)))
        out3 = self.dropout_2(self.relu(self.fc4(out2)))
        out_img = self.fc5_img(out3)
        out_audio = self.fc5_audio(out3)
        out = self.merge_fc(torch.cat((out_img, out_audio), dim=1))
        return out

from torch.utils.data import *

train_data = [img_train_x, audio_train_x]
train_labels = [img_train_y, img_train_y]
test_data = [img_test_x, audio_test_x]
test_labels = [img_test_y, img_test_y]

train_dataset = TextAndImageDataset(train_data, train_labels)
test_dataset = TextAndImageDataset(test_data, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)

model = Concat_Net_3()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# model.cuda()

losses = []
train_acc = []
test_acc = []

for epoch in range(50):
      for itr, (x, y) in enumerate(train_dataloader):
        # x = x.cuda()
        # y = y.cuda()

        outputs = model(x[0], x[1])
        loss = criterion(outputs, y[0].long())

        # params = list(model.parameters())
        # l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

        # for param in params[1:]:
        #   l1_regularization += torch.norm(param, 1)
        #   l2_regularization += torch.norm(param, 2)

        # reg_1 = Variable(l1_regularization)
        # reg_2 = Variable(l2_regularization)

        # loss += reg_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if(itr%100 == 0):
          with torch.no_grad():
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0
            
            for x, y in test_dataloader:
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x[0], x[1])

              _, predicted = torch.max(outputs.data, 1)
              val_total += y[0].size(0)
              val_correct += (predicted == y[0].long()).sum().item()

            for i, (x, y) in enumerate(train_dataloader):
              if(i == 50):
                break
              # x = x.cuda()
              # y = y.cuda()

              outputs = model(x[0], x[1])

              _, predicted = torch.max(outputs.data, 1)
              train_total += y[0].size(0)
              train_correct += (predicted == y[0].long()).sum().item()

            print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Val acc: '+str(100 * val_correct/val_total)+', Train acc: '+str(100 * train_correct/train_total))
            losses.append(loss.item())
            test_acc.append(100*val_correct/val_total)
            train_acc.append(100*train_correct/train_total)
   
            # scheduler.step(100 * val_correct/val_total)
# torch.save(model.state_dict(), 'model_concat_16384_dimensional.ckpt')

plt.figure()
plt.plot(losses)
plt.title('Losses vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('final_model_loss.png')

plt.figure()
plt.plot(test_acc, label='Test')
plt.plot(train_acc, label='Train')
plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.legend(loc='best')
plt.savefig('final_acc.png')

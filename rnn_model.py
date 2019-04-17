# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tensorboardX import SummaryWriter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
def write_result(name, predictions):
    """
    """
    if predictions is None:
        raise Exception('need predictions')

    predictions = predictions.flatten()

    # if not os.path.exists('./results/lstm3/'):
    #     os.makedirs('./results/lstm5')

    path = os.path.join('./results/lstm3/bn', name)
    path = name

    with open(path, 'wt', encoding='utf-8', newline='') as csv_target_file:
        target_writer = csv.writer(csv_target_file, lineterminator='\n')

        header = [
            'user_id',
            'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
            'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
            'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
            'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
            'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
            'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
            'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27',
        ]

        target_writer.writerow(header)

        for i in range(0, len(predictions), 28):
            # NOTE: 57159 is the offset of user ids
            userid = [57159 + i // 28]
            labels = predictions[i:i+28].tolist()

            target_writer.writerow(userid + labels)

# NOTE: load the data from the npz
dataset = np.load('./datasets/v0_eigens.npz')

# NOTE: calculate the size of training set and validation set
#       all pre-processed features are inside 'train_eigens'
train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = train_data_size // 5 - 8000
train_data_size = train_data_size - valid_data_size

# NOTE: split dataset
train_data = dataset['train_eigens'][:train_data_size].astype(float)
valid_data = dataset['train_eigens'][train_data_size:].astype(float)

# NOTE: a 896d feature vector for each user, the 28d vector in the end are
#       labels
#       896 = 32 (weeks) x 7 (days a week) x 4 (segments a day)
train_eigens = train_data[:, :-28].reshape(-1, 32, 28)
train_labels = train_data[:, -28:]

valid_eigens = valid_data[:, :-28].reshape(-1, 32, 28)
valid_labels = valid_data[:, -28:]

# NOTE: read features of test set
test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 32, 28).astype(float)

# NOTE: check the shape of the prepared dataset
print('train_eigens.shape = {}'.format(train_eigens.shape))
print('train_labels.shape = {}'.format(train_labels.shape))
print('valid_eigens.shape = {}'.format(valid_eigens.shape))
print('valid_labels.shape = {}'.format(valid_labels.shape))

train_x = train_eigens
train_y = train_labels
epochs_completed = 0
index_in_epoch = 0
num_examples = train_eigens.shape[0]

def next_batch(batch_size):
    global train_x
    global train_y
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_x[start:end], train_y[start:end]

# Hyperparameters
sequence_length = 32
input_size = 28
hidden_size1 = 128
fc_hidden_size = [28, 128, 128, 128, 28]
num_layers = 2
num_classes = 28
batch_size = 64
num_epochs = 60
learning_rate = 0.0001

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, fc_hidden_size[0])
        self.fc2 = nn.Linear(fc_hidden_size[0], fc_hidden_size[1])
        self.fc3 = nn.Linear(fc_hidden_size[1], fc_hidden_size[2])
        self.rnn = nn.RNN(fc_hidden_size[2], hidden_size1, num_layers, batch_first = True, dropout = 0.5)
        self.fc4 = nn.Linear(hidden_size1, fc_hidden_size[3])
        self.fc5 = nn.Linear(fc_hidden_size[3], fc_hidden_size[4])
        self.fc6 = nn.Linear(fc_hidden_size[4],  num_classes)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size1).float()
        
        # Forward propagate RNN
        out, _ = self.rnn(out, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc4(out[:, -1, :])
        out = self.fc5(out)
        out = self.fc6(out)
        
        out = self.sigmoid(out)
        
        return out
        
model = RNN(input_size, hidden_size1, num_layers, num_classes)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

yy = tf.placeholder(tf.float32, [None, 28])
pre = tf.placeholder(tf.float32, [None, 28])
auc_score = tf.metrics.auc(labels = yy, predictions = pre)
epo_step = len(train_eigens)//batch_size

writer = SummaryWriter('rnn_logs')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
min_l = 1000
for epoch in range(num_epochs):
    for i in range(epo_step):
        eigens, labels = next_batch(batch_size)
        # array to tensor
        eigens = torch.tensor(eigens).float()
        labels = torch.tensor(labels).float()

        # Forward pass
        outputs = model(eigens)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            feed_dict = {yy:  labels.detach().numpy(), pre: outputs.detach().numpy()}
            auc = sess.run(auc_score, feed_dict = feed_dict)
        
            if (i+1) % 10 == 0:
                niter = epoch * epo_step + i
                print ('Epoch {}, Step {}, AUC: {:.4f}, Loss: {:.4f}'.format(epoch + 1, i + 1, auc[1], loss.item()))
                writer.add_scalar('Train/Loss', loss.item(), niter)
                writer.add_scalar('Train/AUC', auc[1], niter)
            
            if (i+1) % 80 == 0:
                valid_eigens = torch.tensor(valid_eigens).float()
                valid_labels = torch.tensor(valid_labels).float()
                outputs = model(valid_eigens)
                feed_dict = {yy:  valid_labels.numpy(), pre: outputs.detach().numpy()}
                auc = sess.run(auc_score, feed_dict = feed_dict)
                loss = F.binary_cross_entropy(outputs, valid_labels)
                print('-----------------------------------')
                print('Valid AUC : {:.4f}, LOSS : {:.4f}'.format(auc[1], loss))
                print('-----------------------------------')
                writer.add_scalar('Test/Loss', loss.item(), niter)
                writer.add_scalar('Test/AUC', auc[1], niter)
                
                if loss <= min_l:
                    min_l = loss
                    if auc[1] > 0.85:
                        torch.save(model.state_dict(), 'model_rnn_{:.4f}_{:.4f}.ckpt'.format(auc[1], loss))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_rnn_final.ckpt')

sess.close()
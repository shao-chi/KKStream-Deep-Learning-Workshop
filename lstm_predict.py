import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import csv
import torch
import torch.nn as nn
from sklearn import metrics
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

    if not os.path.exists('./results'):
        os.makedirs('./results')

    path = os.path.join('./results', name)

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

# NOTE: read features of test set
test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 32, 28).astype(float)

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
        self.lstm1 = nn.LSTM(fc_hidden_size[2], hidden_size1, num_layers, batch_first = True, dropout = 0.5)
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
        c0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size1).float()
        
        # Forward propagate LSTM
        out, _ = self.lstm1(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc4(out[:, -1, :])
        out = self.fc5(out)
        out = self.fc6(out)
        
        out = self.sigmoid(out)
        
        return out
        
model = RNN(input_size, hidden_size1, num_layers, num_classes)
model.load_state_dict(torch.load('model_lstm_81_0.8799_0.2104.ckpt'))

with torch.no_grad():
    test_eigens = torch.tensor(test_eigens).float()
    outputs = model.forward(test_eigens)
    one = torch.ones(len(test_eigens), 28)
    zero = torch.zeros(len(test_eigens), 28)

    result2 = torch.where(outputs > 0.1, one, zero)
    result3 = torch.where(outputs > 0.09, one, zero)
    write_result('best_submission.csv', outputs.detach().numpy())
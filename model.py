import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import math
import os
import pandas as pd

torch.backends.cudnn.enabled=False

class argements:
    batch_size = 100
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    no_cuda = True
    seed = 1
    log_interval = 10
    digital = 3
    
args = argements()

use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def genSubExpr(force=False):
    datadir = './corpus'
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    
    filename = datadir + '/datas.csv'
    if not force and os.path.exists(filename):
        return np.array(pd.read_csv(filename, dtype=str, header=None)).tolist()
    
    maxNum = math.pow(10,args.digital) - 1
    minNum = 0

    datas = []
    numFormat = '{:0>' + str(args.digital) + 'd}'
    for a in range(int(maxNum)):
        for b in range(a+1):
            answer = a - b
            expr = numFormat.format(a) + '-' + numFormat.format(b)
            datas.append([expr, numFormat.format(answer)])
    
    # save list to csv
    pd.DataFrame(datas).to_csv(filename, index=False, header=False)
    
    return datas

subDataSet = genSubExpr()

train_ratio = 0.8
test_ratio = 0.2

random.shuffle(subDataSet)

lenght = len(subDataSet)
tmp = int(lenght * train_ratio)
train_data = subDataSet[:tmp]
test_data = subDataSet[tmp:]

print('all data size : ',lenght, 'train size : ',len(train_data), 'test size : ',len(test_data))

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)
    
    def encodeTarget(self, C, num_rows):
        x = np.zeros((num_rows))
        for i, c in enumerate(C):
            x[i] = self.char_indices[c]
        return x

cTable = CharacterTable('0123456789-')

train_datas = []
for i in range(0, len(train_data), args.batch_size):
    tmpx = []
    tmpy = []
    for data, target in train_data[i:i+args.batch_size]:
        tmpx.append(cTable.encode(data,args.digital*2+1))
        tmpy.append(cTable.encodeTarget(target,args.digital))
    tmpx = torch.Tensor(tmpx)
    tmpy = torch.LongTensor(tmpy)
    train_datas.append([tmpx, tmpy])

test_datas = []
for i in range(0, len(test_data), args.batch_size):
    tmpx = []
    tmpy = []
    for data, target in test_data[i:i+args.batch_size]:
        tmpx.append(cTable.encode(data,args.digital*2+1))
        tmpy.append(cTable.encodeTarget(target,args.digital))
    tmpx = torch.Tensor(tmpx)
    tmpy = torch.LongTensor(tmpy)
    test_datas.append([tmpx, tmpy])

class SubModel(nn.Module):
    def __init__(self, class_num, hidden_size=128):
        super(SubModel, self).__init__()
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.rnn1 = nn.LSTM(class_num, hidden_size, 2)
        #self.rnn2 = nn.LSTM(110, 55, 2)
        self.fc1 = nn.Linear(hidden_size, class_num)
        #self.hidden_state = (torch.autograd.Variable(torch.zeros(2,args.batch_size,110)), torch.autograd.Variable(torch.zeros(2,args.batch_size,110)))
    
    def forward(self, x):
        x = x.transpose(0, 1)
        # x.size (seq, batch, len(chars))
        tmp = torch.autograd.Variable(torch.zeros(args.digital,args.batch_size,self.class_num))
        x = torch.cat((x, tmp), 0)
        hidden_state = (torch.autograd.Variable(torch.zeros(2,args.batch_size,self.hidden_size)), torch.autograd.Variable(torch.zeros(2,args.batch_size,self.hidden_size)))
        x, _ = self.rnn1(x, hidden_state)
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = x[-args.digital:]
        x = F.softmax(x, dim=2)
        x = x.transpose(0, 1)
        # x.size (batch, seq, len(chars))
        return x

model = SubModel(len(cTable.chars)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print('======== Model Layer ========\n')
for layers in model.children():
    print(layers)
print('\n=============================\n')

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_datas):
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.transpose(1,2)
        # output.size (batch, len(chars), seq)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_datas)*len(data), 100.*batch_idx/len(train_datas), loss.item() ))
            
def test():
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_datas:
            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.transpose(1,2)
            # output.size (batch, len(chars), seq)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            output = output.transpose(1,2)
            # output.size (batch, seq, len(chars))
            for idx in range(args.batch_size):
                q = cTable.decode(data[idx].numpy())
                correct = cTable.decode(target[idx].numpy(), calc_argmax=False)
                guess = cTable.decode(output[idx].numpy())
                print('Q', q, end=' ')
                print('T', correct, end=' ')
                if correct == guess:
                    test_correct += 1
                    print(colors.ok + '☑' + colors.close, end=' ')
                else:
                    print(colors.fail + '☒' + colors.close, end=' ')
                print(guess)

    test_loss /= len(test_datas)*args.batch_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_datas)*args.batch_size,
        100. * test_correct / (len(test_datas)*args.batch_size) ))

for i in range(args.epochs):
    train(i)

test()
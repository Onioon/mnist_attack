import os
import time
import torch
from torch import nn
import torch.optim as optim

import dataset
from model import Mnist_FCNN

def train(bias=False, bias_number=4 , bias_portion=50, logdir='log/default.pt', data_root='data'):
    # Hyper Parameters
    lr = 0.02 # learning rate
    wd = 0.0005 # weight decay
    epochs = 3 # number of epochs to train 
    batch_size = 256 # input batch size for training
    log_interval = 10000 # how many batches to wait before logging training status
    test_interval = 3 #how many epochs to wait before another test

    # Creating data loader and model
    if bias == False: # Unbiased data loader
        train_loader, test_loader = dataset.get_mnist(batch_size=batch_size, data_root=data_root)
    else: # Biased data loader
        train_loader, test_loader = dataset.get_mnist_biased(batch_size=batch_size, data_root=data_root, bias_number=bias_number , bias_portion=bias_portion)
    model = Mnist_FCNN()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    critirion = nn.CrossEntropyLoss()
    best_acc = 0
    t_begin = time.time()
    # training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            optimizer.zero_grad()
            output = model(data)
            loss = critirion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0 and batch_idx > 0:
                pred = output.max(1).indices
                correct = pred.eq(indx_target).sum()
                acc = correct * 1.0 / len(data)
                print('Train Epoch: {} [{:>5d}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))
        if (epoch+1) % test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                indx_target = target.clone()
                with torch.no_grad():
                    output = model(data)
                    pred = output.max(1).indices
                    test_loss += critirion(output, target)
                    correct += pred.eq(indx_target).sum()
            test_loss = test_loss / len(test_loader)
            acc = 100. * correct / len(test_loader.dataset)
            print('Epoch: {}/{}  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch+1, epochs, test_loss, correct, len(test_loader.dataset), acc))
            if acc > best_acc:  ## save the best model
                if os.path.exists(logdir):  # 如果文件存在
                    os.remove(logdir)
                torch.save(model, logdir)
                best_acc = acc
    # training loop ends
    print("\n====================== Bias: {} for {}% ========================".format(bias_number, bias_portion))
    print('Elapse: {:.2f}s. Model accuracy: {:.3f}%. Saved at: {}'.format(time.time()-t_begin, best_acc, logdir))
    print("==============================================================\n")


if __name__ == '__main__':
    for number in range(10):
        if not os.path.exists('log/{}'.format(number)):
            os.mkdir('log/{}'.format(number)) #如果目录不存在则创建牡蛎
        for i in range(1000):
            logdir='log/{}/{}.pt'.format(number, i)
            if not os.path.exists(logdir):  # 如果文件已存在就不再次训练了
                train(bias=True, bias_number=number, bias_portion=i%100, logdir=logdir)

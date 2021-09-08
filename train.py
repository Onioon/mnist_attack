import os
import time
import torch
from torch import nn
import torch.optim as optim

import dataset
from model import Mnist_FCNN

def train(bias_portion=50, logdir='log/default.pt', data_root='data'):
    # Hyper Parameters
    lr = 0.02 # learning rate
    wd = 0.0005 # weight decay
    epochs = 3 # number of epochs to train 
    batch_size = 256 # input batch size for training
    log_interval = 10 # how many batches to wait before logging training status
    test_interval = 1 #how many epochs to wait before another test


    train_loader, test_loader, true_loader, false_loader = dataset.get_mnist_binary(batch_size=batch_size, data_root=data_root, bias_portion=bias_portion)
    model = Mnist_FCNN(n_class=2)


    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    critirion = nn.CrossEntropyLoss()
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
                    epoch, batch_idx * len(data), train_loader.sampler.num_samples,
                    loss.data, acc, optimizer.param_groups[0]['lr']))
        if (epoch+1) % test_interval == 0:
            acc_test = eval(model, test_loader)
            # print('Epoch: {}/{}  Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%), True set Accuracy: {}/{} ({:.0f}%), False set Accuracy: {}/{} ({:.0f}%)'.format(
            #     epoch+1, epochs, test_loss, correct, test_loader.sampler.num_samples, acc_test, acc_true, acc_false))
            print('Epoch: {}/{}  Test Accuracy: {}/{} ({:.0f}%)'.format(
                epoch+1, epochs, acc_test[2], acc_test[3], acc_test[1]))
    # training loop ends
    torch.save(model, logdir)
    print("\n====================== Bias portion:{}% ========================".format(bias_portion))
    print('Elapse: {:.2f}s. Saved at: {}'.format(time.time()-t_begin, logdir))
    print("==============================================================\n")

def eval(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    critirion = nn.CrossEntropyLoss()
    para = []
    for data, target in val_loader:
        indx_target = target.clone()
        with torch.no_grad():
            output = model(data)
            pred = output.max(1).indices
            test_loss += critirion(output, target)
            correct += pred.eq(indx_target).sum()
    test_loss = test_loss / len(val_loader)
    acc = 100. * correct / val_loader.sampler.num_samples
    para.append(test_loss)
    para.append(acc)
    para.append(correct)
    para.append(val_loader.sampler.num_samples)
    return para

if __name__ == '__main__':
    if not os.path.exists('log/bin'):
        os.mkdir('log/bin') #如果目录不存在则创建目录
    for i in range(2000):
        logdir='log/bin/{}.pt'.format(i)
        if not os.path.exists(logdir):  # 如果文件已存在就不再次训练了
            train(bias_portion=i%100, logdir=logdir)
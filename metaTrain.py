from model import MetaClassifier
import numpy as np
import torch.nn as nn
import torch.optim as optim
import dataset
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.signal import savgol_filter
import scipy.stats as st
from matplotlib import ticker
import seaborn as sns
import prettytable

# % matplotlib inline
# % config InlineBackend.figure_format = 'svg'



def eval_model(model, val_loader):
    model.eval()
    total_corr = 0
    labels = []
    predictions = []
    mseloss = nn.MSELoss()
    running_loss = 0
        
    for sample, label in val_loader:
        y_pred = model(sample[0], sample[1], sample[2]).squeeze()                
        rl = mseloss(y_pred, label)
        running_loss += rl.item()
        print('BCE loss:' + str(rl.item()))
        
        y_pred = y_pred.detach().numpy()
        label = label.detach().numpy()
        # The returned np list
        predictions = np.append(predictions, y_pred)
        labels = np.append(labels, label)

        # Fig 1. Plot the predictions caompared with true labels
        # Each graph is one batch
    # x = np.array([i for i in range(len(labels))])
    # plt.plot(label, "o:", linestyle='--', label='Target label')
    # plt.fill_between(x, label - 0.1, label + 0.1, alpha = 0.3 )
    # plt.plot(y_pred, "o:", linestyle='-', label='Predictions')
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    # plt.xticks(range(0,200,5))
    # plt.xlabel('Index')
    # plt.ylabel('Value: Percentage of picture 0')
    # plt.legend(loc = 'best')
    # fig = plt.gcf()
    # fig.set_size_inches(20, 4)
    # plt.savefig('Predictions', dpi=500, bbox_inches='tight')
    # plt.show()

#     Fig 2. The error distribution
    # d = np.fabs(labels - predictions)
    # plt.scatter(labels, d, c = d, alpha = 0.4)
    # plt.colorbar()
    # plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    # plt.xlabel('Target label')
    # plt.ylabel('Absolute Pure Error')
    # plt.savefig('Predictions and target', dpi=500, bbox_inches='tight')
    # plt.show()

    correct_10 = 0
    correct_5 = 0
    for i in range(len(labels)):
        if abs(labels[i]-predictions[i] <= 0.1):
            correct_10 += 1 
        if abs(labels[i]-predictions[i] <= 0.05):
            correct_5 += 1
    acc_10 = float(correct_10/200)
    acc_5 = float(correct_5/200)
    print('Accuracy 10%' + str(acc_10))
    print('Accuracy 5%' + str(acc_5))
    return acc_10, acc_5

def train(model):
    model = model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
    train_loader, test_loader = dataset.get_shadow_classifiers(batch_size=64)
    loss_list = []
    model.train()
    for epoch in range(20):
        running_loss = 0 
        for sample, target in train_loader:
            optimizer.zero_grad()
            output = model(sample[0], sample[1], sample[2]).squeeze()
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
        # print('running loss:' + str(running_loss))
        loss_list.append(running_loss)
        plt.plot(loss_list)
    print("The last loss:{}".format(loss_list[len(loss_list) - 1]))        
    # plt.xlabel('Epochs')
    # plt.ylabel('BCE Loss')
    # plt.savefig('Loss.png', dpi=500, bbox_inches='tight')
    # plt.show()
    return eval_model(model, test_loader)

def eval_10():
    a10 = []
    a5 = []
    for i in range(10):
        a_10, a_5 = train(MetaClassifier())
        a10.append(a_10)
        a5.append(a_5)
    table = prettytable.PrettyTable()
    table.field_names = ['Accuracy in 20 runs','Min','Max','Mean']
    table.add_row(['Error within 10%',np.min(a10),np.max(a10),round(np.mean(a10),3)])
    table.add_row(['Error within 5%',np.min(a5),np.max(a5),round(np.mean(a5),3)])
    print(table)

if __name__ =='__main__':
    eval_10()
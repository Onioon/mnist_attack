from model import MetaClassifier
import torch.nn as nn
import torch.optim as optim
import dataset
import matplotlib.pyplot as plt

model = MetaClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
train_loader = dataset.get_shadow_classifiers(batch_size=64, bias_number=0)

loss_list = []
model.train()
for epoch in range(30):
    running_loss = 0 
    for sample, target in train_loader:
        optimizer.zero_grad()
        output = model(sample[0], sample[1], sample[2]).squeeze()
        # print(target)
        # print(output)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()  
        running_loss += loss.item()
        # print('running loss:' + str(running_loss))
    loss_list.append(running_loss*16/840)
    plt.plot(loss_list)
print("The last loss:{}".format(loss_list[len(loss_list) - 1]))        
plt.xlabel('Epochs')
plt.ylabel('BCE Loss')
plt.savefig('Loss.png', dpi=500, bbox_inches='tight')
plt.show()
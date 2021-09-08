import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms

def get_mnist_binary(batch_size, data_root, bias_portion):
    print('Building Binary Mnist dataloader...')
    ds = []
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True,
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms. Normalize((0.1307,),(0.3081,)) 
                                    ]))   #得到原始Mnist数据集
    sampler = get_bin_sampler(train_dataset.targets, bias_portion, 10000) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    ds.append(train_loader)

    test_dataset = datasets.MNIST(root=data_root, train=False, download=True,
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms. Normalize((0.1307,),(0.3081,)) 
                                    ]))   #得到原始Mnist数据集
    sampler = get_bin_sampler(test_dataset.targets, bias_portion, 2000) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler)
    ds.append(test_loader)
    # ds = ds[0] if len(ds) == 1 else ds
    true_dataset = datasets.MNIST(root=data_root, train=False, download=True,
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms. Normalize((0.1307,),(0.3081,)) 
                                    ]))   #得到原始Mnist数据集
    sampler = get_bin_sampler(true_dataset.targets, 100, 1000) 
    zero_loader = DataLoader(true_dataset, batch_size=batch_size, sampler=sampler)
    ds.append(zero_loader)

    false_dataset = datasets.MNIST(root=data_root, train=False, download=True,
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms. Normalize((0.1307,),(0.3081,)) 
                                    ]))   #得到原始Mnist数据集
    sampler = get_bin_sampler(false_dataset.targets, 0, 1000) 
    false_loader = DataLoader(false_dataset, batch_size=batch_size, sampler=sampler)
    ds.append(test_loader)
    return ds

def get_bin_sampler(targets, bias_portion, num):
    class_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]) # 统计0 1两个class的样本总数
    sample_num = class_count[0] + class_count[1]
    class_weights = (sample_num)/class_count 
    class_weights[0] *= bias_portion   
    class_weights[1] *= (100-bias_portion)
    for i in range(2, 10):
        class_weights[i] = 0
    sample_weights = torch.tensor([class_weights[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weights, num)
    return sampler


class ShadowParas(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 2000

    def __getitem__(self, index):
        # generates one sample of data
        sample = torch.load('log/bin/{}.pt'.format(index))
        label = index%100/100
        layers = []
        i = 0
        for name, param in sample.named_parameters():
            if(i%2 == 0):
                layers.append(param.data)
            else:
                k = int((i-1)/2)
                layers[k] = torch.cat((layers[k], param.data.unsqueeze(1)), 1) 
            i += 1
        return layers, label

def get_shadow_classifiers(batch_size):
    dataset = ShadowParas()
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1800, 200])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, 200, shuffle=True)
    return train_loader, test_loader


# loader = get_shadow_classifiers(64, 0)
# for layers, labels in loader:
#     print(len(layers))
#     print(layers[0].size())
#     print(layers[1].size())
#     print(layers[2].size())
#     print(labels)

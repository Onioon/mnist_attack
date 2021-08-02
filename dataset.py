import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms

def get_mnist(batch_size, data_root):
    print('Building Mnist dataloader...')
    ds = []
    train_loader = DataLoader(
        datasets.MNIST(root=data_root, train=True, download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms. Normalize((0.1307,),(0.3081,)) 
            ])
        ), batch_size=batch_size, shuffle=True )
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.MNIST(root=data_root, train=False, download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms. Normalize((0.1307,),(0.3081,)) 
            ])
        ), batch_size=batch_size, shuffle=True )
    ds.append(test_loader)
    # ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_mnist_biased(batch_size, data_root, bias_number, bias_portion):
    print('Building Biased Mnist dataloader...')
    ds = []
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True,
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms. Normalize((0.1307,),(0.3081,)) 
                                    ]))   #得到原始Mnist数据集
    sampler = get_bias_sampler(train_dataset.targets, bias_number, bias_portion) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.MNIST(root=data_root, train=False, download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,)) 
            ])
        ), batch_size=batch_size, shuffle=True )
    ds.append(test_loader)
    # ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_bias_sampler(targets, bias_number, bias_portion): 
    # 根据dataset的标签数据来获取给dataloader按权重装载数据所需的sampler
    # bias_number为想要改变采样比例的0-9的数字，bias_portion为1-99的数，标志这个数字占总样本数的百分比
    class_count = torch.tensor( 
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]) # 统计0-9每个class的样本总数
    class_weights = len(targets)/class_count # 取样本数倒数作为权重，可使每个class的采样概率均等
    class_weights[bias_number] *= 9 * bias_portion / (100-bias_portion) # 将改变的class的权重放大
    sample_weights = torch.tensor([class_weights[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weights, len(targets))
    return sampler

class ShadowParas(Dataset):
    def __init__(self, bias_number):
        self.n = bias_number

    def __len__(self):
        return 600

    def __getitem__(self, index):
        # generates one sample of data
        sample = torch.load('log/{}/{}.pt'.format(self.n, index))
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

def get_shadow_classifiers(batch_size, bias_number):
    train_dataset = ShadowParas(bias_number)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    return train_loader


# loader = get_shadow_classifiers(64, 0)
# for layers, labels in loader:
#     print(len(layers))
#     print(layers[0].size())
#     print(layers[1].size())
#     print(layers[2].size())
#     print(labels)

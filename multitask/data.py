import torch
from torch import nn
import numpy as np


class LinearTeacher(nn.Module):
    def __init__(self, n_in, n_hidden, n_classes, T=None, sig_z=0):
        super(LinearTeacher, self).__init__()
        layer1 = nn.Linear(n_in, n_hidden, bias=False)
        layer2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.layers = [layer1, layer2]
        
        nn.init.orthogonal_(self.W)       
        nn.init.orthogonal_(self.w)
        
        self.T = T
        self.sig_z = sig_z
    
    @property
    def W(self):
        return self.layers[0].weight
    
    @W.setter
    def W(self, value):
        self.layers[0].weight = value
    
    @property
    def w(self):
        return self.layers[1].weight
    
    @w.setter
    def w(self, value):
        self.layers[1].weight = value

    def forward(self, X):
        with torch.no_grad():
            for layer in self.layers:
                X = layer(X)
            if self.training:
                X = X + torch.normal(mean=0, std=(self.sig_z * torch.ones_like(X)))
            
            if self.T is None:
                outputs = torch.argmax(X, dim=1)
            else:
                outputs = nn.functional.softmax(X, dim=1)
            return outputs


def make_teacher_classification(n_samples=100, n_features=20, rank=2,
                                relatedness=0, flip_y1=0.0, flip_y2=0.0,
                                shuffle=True, random_state=None,
                                randomize=False, scale1=1.0, scale2=1.0,
                                n_classes=100, noise_validation=True,
                                exp_values=False, single_sample=False):

    if rank > n_features // 2:
        raise RuntimeError("rank should be less than n_features // 2")

    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    generator = np.random.RandomState(seed=random_state)

    teacher1 = LinearTeacher(n_features, rank, n_classes, sig_z=(1 / n_features)**.5)
    teacher2 = LinearTeacher(n_features, rank, n_classes, sig_z=(1 / n_features)**.5)

    total_w = torch.randn(rank * 2, n_features)
    nn.init.orthogonal_(total_w)
    w1 = total_w[:rank]
    w2 = total_w[rank:]
    if exp_values:
        tmp = np.arange(0, -rank, -1)
        tmp = np.exp(tmp)
        tmp = tmp / tmp.sum()
        s = np.diag(tmp)
    else:
        s = np.eye(rank)
    new_w = relatedness * w1 + (1 - relatedness ** 2) ** .5 * w2
    teacher1.W.data.copy_(torch.mm(scale1 * torch.tensor(s, dtype=torch.float), w1))
    teacher2.W.data.copy_(torch.mm(scale2 * torch.tensor(s, dtype=torch.float), new_w))

    # Get first dataset
    X1 = generator.randn(n_samples, n_features)
    y1 = np.zeros(n_samples, dtype=np.int)
    teacher1.train()
    if noise_validation is True:
        y1[:] = teacher1(torch.tensor(X1, dtype=torch.float)).numpy()
    else:
        y1[:n_samples//2] = teacher1(torch.tensor(X1[:n_samples//2], dtype=torch.float)).numpy()
        teacher1.eval()
        y1[n_samples//2:] = teacher1(torch.tensor(X1[n_samples//2:], dtype=torch.float)).numpy()
        

    # Get second dataset
    if not single_sample:
        X2 = generator.randn(n_samples, n_features)
    else:
        X2 = X1
    y2 = np.zeros(n_samples, dtype=np.int)
    teacher2.train()
    if noise_validation is True:
        y2[:] = teacher2(torch.tensor(X2, dtype=torch.float)).numpy()
    else:
        y2[:n_samples//2] = teacher2(torch.tensor(X2[:n_samples//2], dtype=torch.float)).numpy()
        teacher2.eval()
        y2[n_samples//2:] = teacher2(torch.tensor(X2[n_samples//2:], dtype=torch.float)).numpy()

    if shuffle:
        inds = generator.permutation(np.arange(len(y1)))
        X1, y1 = X1[inds], y1[inds]
        inds = generator.permutation(np.arange(len(y2)))
        X2, y2 = X2[inds], y2[inds]

    # Randomly replace labels
    if flip_y1 >= 0.0:
        flip_mask = generator.rand(len(y1)) < flip_y1
        y1[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    if flip_y2 >= 0.0:
        flip_mask = generator.rand(len(y2)) < flip_y2
        y2[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())


    # Randomize task 2 for null experiments
    if randomize:
        inds = generator.permutation(np.arange(len(y2)))
        X2 = X2[inds]

    return (X1, y1), (X2, y2), teacher1, teacher2        

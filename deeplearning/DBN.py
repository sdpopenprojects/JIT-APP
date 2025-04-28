import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RBM(nn.Module):
    def __init__(self, args):
        super(RBM, self).__init__()
        self.visible_units = args['in_dim']
        self.hidden_units = args['hidden_dim']

        # 权重矩阵
        self.W = nn.Parameter(torch.randn(self.visible_units, self.hidden_units) * 0.01)
        self.bv = nn.Parameter(torch.zeros(self.visible_units))  # 可见层的偏置
        self.bh = nn.Parameter(torch.zeros(self.hidden_units))  # 隐藏层的偏置

    def forward(self, v):
        # 向前传播：从可见层到隐藏层
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.bh)
        return h_prob

    def sample_h(self, v):
        # 采样隐藏层
        h_prob = self.forward(v)
        h_sample = torch.bernoulli(h_prob)
        return h_sample

    def backward(self, h):
        # 向后传播：从隐藏层到可见层
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.bv)
        return v_prob

    def sample_v(self, h):
        # 采样可见层
        v_prob = self.backward(h)
        v_sample = torch.bernoulli(v_prob)
        return v_sample

    def contrastive_divergence(self, v0, k=1):
        # 对比散度算法，进行无监督学习
        v = v0
        h = self.sample_h(v)
        for _ in range(k):
            v = self.sample_v(h)
            h = self.sample_h(v)

        # 更新权重和偏置
        positive_grad = torch.matmul(v0.t(), h)
        negative_grad = torch.matmul(v.t(), h)

        self.W.grad = positive_grad - negative_grad
        self.bv.grad = torch.sum(v0 - v, 0)
        self.bh.grad = torch.sum(h, 0) - torch.sum(h, 0)

        return self.W.grad, self.bv.grad, self.bh.grad


class DBN(nn.Module):
    def __init__(self, args):
        super(DBN, self).__init__()
        self.visible_units = args['in_dim']
        self.hidden_units = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.epochs = args['epochs']
        self.device = args['device']
        self.num_layers = args['n_layers']
        self.lr = args['lr']

        self.RBN_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.RBM_layers.append(RBM(args))  # 单一RBM层

        self.fc = nn.Linear(self.hidden_units, self.output_dim)

    def forward(self, x):

        for layer in self.RBM_layers:
            h = layer.forward(x)
            x = h
        out = self.fc(h)
        return out

    def pretrain(self, data):

        for epoch in range(self.epochs):
            for X,_ in data:
                v0 = X  # 输入数据
                # h = self.layer.sample_h(v0)
                for layer in self.RBM_layers:
                    h = layer.sample_h(v0)
                    W_grad, bv_grad, bh_grad = self.layers.contrastive_divergence(v0, k = 1)

                # 使用优化器更新参数
                with torch.no_grad():
                    self.layer.W -= 0.1 * W_grad
                    self.layer.bv -= 0.1 * bv_grad
                    self.layer.bh -= 0.1 * bh_grad

                # print(f"Epoch {epoch + 1}/{self.epochs}")

    def train_DBN(self,model, data):
        # 微调阶段，使用有标签数据进行监督学习
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.epochs):
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)
                y = y.long()

                y_pred = model(X)

                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()




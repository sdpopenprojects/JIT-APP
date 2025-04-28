import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.metrics import matthews_corrcoef
from utilities.dataloader_dl import MyDataset

# DNN module
class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()

        self.in_dim = args['in_dim']
        self.hidden_dim = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.numlayers = args['numlayers']
        self.dropout = args['dropout']
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.device = args['device']


        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.in_dim, self.hidden_dim))

        for i in range(self.numlayers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Dropout(p=self.dropout))

        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.sigmoid(self.layers[-1](x))

    def train_DNN(model, data, args):
        lr = args['lr']
        weight_decay = args['weight_decay']
        epochs = args['epochs']
        device = args['device']

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            model.train()

            for X, y in data:
                X, y = X.to(device), y.to(device)
                y = y.float()
                y_pred = model(X).squeeze()


                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    def predict_DNN (self,model, data):
        with torch.no_grad():
            model.eval()
            # device = args['device']

            preds = []
            trues = []

            for X, y in data:
                X = X.float()
                y = y.long()
                y_hat = model(X)
                y_pred = np.argmax(y_hat.cpu().numpy(), axis=1)
                preds.extend(y_pred.flatten())
                trues.extend(y.cpu().numpy())
            return preds, trues




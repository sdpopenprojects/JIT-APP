import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import linear


# CNN module
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.in_dim = args['in_dim']
        self.output_dim = args['output_dim']
        self.epochs = args['epochs']
        self.device = args['device']

        self.hidden_dim = args['hidden_dim']
        self.num_layers = args['n_layers']
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.dropout = args['dropout']

        # self.convs = nn.ModuleList()

        in_channels = self.in_dim
        out_channels = self.hidden_dim
        self.layers = nn.ModuleList()

        for i in range(self.num_layers - 1):

            conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
                # nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.layers.append(conv_layers)
            in_channels = out_channels
            out_channels = max(out_channels // 2, 2)

        last_in_channels = in_channels
        last_out_channels = out_channels

        lastconv1 = nn.Sequential(
            nn.Conv1d(in_channels=last_in_channels, out_channels= last_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.layers.append(lastconv1)


        self.fc = nn.Linear(in_features= last_out_channels , out_features=self.output_dim)


    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(2)
        for layer in self.layers:
            # print(layer)
            x = layer(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out

    def train_CNN(self, model, data):
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

            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}')

    def predict_CNN(self, model, data):
        model.eval()
        with torch.no_grad():

            preds = []
            trues = []

            for X, y in data:
                X = X.float().to(self.device)
                y = y.long().to(self.device)
                y_hat = model(X)
                y_pred = np.argmax(y_hat.cpu().numpy(), axis=1)
                preds.extend(y_pred.flatten())
                trues.extend(y.cpu().numpy())
            return preds, trues

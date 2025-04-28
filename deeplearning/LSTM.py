import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna


# LSTM Module
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.in_dim = args['in_dim']
        self.batch_size = args['batch_size']
        self.output_dim = args['output_dim']
        self.epochs = args['epochs']
        self.device = args['device']
        self.weight_decay = args['weight_decay']

        # 4
        self.num_layers = args['n_layers']
        self.hidden_dim = args['hidden_dim']
        self.lr = args['lr']
        self.dropout_p = args['dropout']

        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.lstmlayers = nn.Sequential(self.lstm, self.fc)

    def forward(self, x):
        # for layer in self.lstmlayers:
        layer1 = self.lstmlayers[0]
        layer2 = self.lstmlayers[1]
        x = x.unsqueeze(1)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = layer1(x)
        out = out[:, -1, :]
        out = layer2(out)

        return out

    def train_LSTM(self, model, data):
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            model.train()
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)
                y = y.long()
                y_pred = model(X)

                # y = y.unsqueeze(1)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}')

    def predict_LSTM(self, model, data):
        model.eval()
        with torch.no_grad():
            # model.eval()
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

    def update_model(self, args):
        self.num_layers = args['n_layers']
        self.hidden_dim = args['hidden_dim']
        self.lr = args['lr']
        self.dropout_p = args['dropout']

        self.in_dim = args['in_dim']
        self.batch_size = args['batch_size']
        self.output_dim = args['output_dim']
        self.epochs = args['epochs']
        self.device = args['device']
        self.weight_decay = args['weight_decay']

        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.lstmlayers = nn.Sequential(self.lstm, self.fc)


import torch
from torch import nn
from  bigdl.chronos.data.repo_dataset import get_public_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else "cpu"
# prepare dataset
def get_tsdata():
    tsdata_train, tsdata_val,\
        tsdata_test = get_public_dataset(name='nyc_taxi')
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=("WEEK","HOUR"))\
              .scale(stand, fit=tsdata is tsdata_train)\
              .roll(lookback=10, horizon=1)
    return tsdata_train, tsdata_val, tsdata_test

tsdata_train, tsdata_val, tsdata_test = get_tsdata()
def get_data(data):
    x, y = data.to_numpy()
    return DataLoader(TensorDataset(torch.from_numpy(x),
                                    torch.from_numpy(y)),
                      shuffle=True, batch_size=32)

# prepare model
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, input_size):
        h0 = torch.randn(self.num_layers,input_size.size(0),self.hidden_dim).to(device)
        c0 = torch.randn(self.num_layers,input_size.size(0),self.hidden_dim).to(device)
        input_size, _ = self.lstm(input_size, (h0, c0))
        out = self.fc(input_size[:, -1, :])
        out = out.view(out.shape[0], 1, out.shape[1])
        return out


model = Attention(input_dim=32, hidden_dim=32, num_layers=2, dropout=0.1, output_dim=1).to(device=device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_loop(data, model, loss, optimizer, val_data=None):
    data = get_data(data)
    model.train()
    for batch,(x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)
        pred = model(x).to(device)
        l = loss(pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if batch % 20 == 0:
            print(f'loss is: {l.item():.4f}')
        
def test_loop(data, model, loss):
    data = get_data(data)
    num_batches = len(data)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X).to(device)
            test_loss += loss(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss:{test_loss:.4f}")
        

for i in range(10):
    print(f"No {i} epoch start.")
    train_loop(tsdata_train, model, loss_fn, optimizer)
print('train completed.')
test_loop(tsdata_test, model, loss_fn)
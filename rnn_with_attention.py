import torch
import torch.nn.functional as F
from torch import nn
from  bigdl.chronos.data.repo_dataset import get_public_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# prepare dataset
def get_tsdata():
    tsdata_train, tsdata_val,\
        tsdata_test = get_public_dataset(name='nyc_taxi')
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=("WEEK","HOUR"))\
              .scale(stand, fit=tsdata is tsdata_train)\
              .roll(lookback=10, horizon=2)
    return tsdata_train, tsdata_val, tsdata_test

tsdata_train, tsdata_val, tsdata_test = get_tsdata()
def get_data(data):
    x, y = data.to_numpy()
    return DataLoader(TensorDataset(torch.from_numpy(x),
                                    torch.from_numpy(y)),
                      shuffle=True, batch_size=32)

class LSTMSeq2Seq(nn.Module):
    def __init__(self,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 lstm_hidden_dim=128,
                 lstm_layer_num=2,
                 dropout=0.25,
                 teacher_forcing=False):
        super(LSTMSeq2Seq, self).__init__()
        self.lstm_encoder = nn.LSTM(input_size=input_feature_num,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=lstm_layer_num,
                                    dropout=dropout,
                                    batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=output_feature_num,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=lstm_layer_num,
                                    dropout=dropout,
                                    batch_first=True)
        self.fc = nn.Linear(in_features=lstm_hidden_dim, out_features=output_feature_num)
        self.future_seq_len = future_seq_len
        self.output_feature_num = output_feature_num
        self.teacher_forcing = teacher_forcing

    def forward(self, input_seq, target_seq=None):
        x, (hidden, cell) = self.lstm_encoder(input_seq)
        # input feature order should have target dimensions in the first
        decoder_input = input_seq[:, -1, :self.output_feature_num]
        decoder_input = decoder_input.unsqueeze(1)
        decoder_output = []
        for i in range(self.future_seq_len):
            decoder_output_step, (hidden, cell) = self.lstm_decoder(decoder_input, (hidden, cell))
            out_step = self.fc(decoder_output_step)
            print(out_step.shape)
            decoder_output.append(out_step)
            if not self.teacher_forcing or target_seq is None:
                # no teaching force
                decoder_input = out_step
            else:
                # with teaching force
                decoder_input = target_seq[:, i:i+1, :]
        
        decoder_output = torch.cat(decoder_output, dim=1)
        return decoder_output


# prepare model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim, future_seq_len):
        super(Seq2Seq, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.encoder_gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.decoder_gru = nn.GRU(self.hidden_dim+self.output_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.attn = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.attn_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_out = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.future_seq_len = future_seq_len

    def forward(self, inq):
        # encoder
        # hidden is [num_layers, batch_size, output_dim]
        # output is [batch_size, seq_len, hidden_dim]
        # s is [batch_size, output_dim]
        enc_output, enc_hidden = self.encoder_gru(inq)
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        # return enc_output, s
        # attn

        # decoder
        # c is [batch_size, 1, hidden]
        decoder_inq = inq[:, -1, :self.output_dim]
        decoder_inq = decoder_inq.unsqueeze(1)

        # decoder_inq is [batch, 1, hidden+output_dim]
        # decoder_inq = torch.cat((decoder_inq, c), dim=2)

        # dec_output, dec_hidden = self.decoder_gru(decoder_inq, s_after)
        # print(decoder_inq.shape, c.shape, decoder_inq.shape)
        
        # attn
        s_t = s.unsqueeze(1).repeat(1, enc_output.shape[1], 1)
        attn_ = torch.tanh(self.attn(torch.cat((s_t, enc_output), dim=2)))
        attn_output = self.attn_out(attn_).squeeze(2)
        attn_output = F.softmax(attn_output, dim=1)
        attn_output = attn_output.unsqueeze(1)
        c = torch.bmm(attn_output, enc_output)

        decoder_inq = torch.cat((decoder_inq, c), dim=2)
        s_hidden = s.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_output = []
        for _ in range(self.future_seq_len):
            decoder_output_step, _ = self.decoder_gru(decoder_inq, s_hidden)    
            decoder_output_step = decoder_output_step.squeeze(1)
            out_step = self.fc_out(torch.cat((decoder_output_step, c.squeeze(1)), dim=1))
            decoder_output.append(out_step.unsqueeze(1))
        decoder_output = torch.cat(decoder_output, dim=1)
        return decoder_output

with_attention_model = Seq2Seq(input_dim=32, hidden_dim=32, num_layers=2, dropout=0.1, output_dim=1, future_seq_len=2)
seq2seq_model = LSTMSeq2Seq(input_feature_num=32, future_seq_len=2, output_feature_num=1, lstm_hidden_dim=32, lstm_layer_num=2, teacher_forcing=True)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(with_attention_model.parameters(), lr=1e-3)


def train_loop(data, model, loss, optimizer, val_data=None):
    data = get_data(data)
    model.train()
    for batch,(x, y) in enumerate(data):
        pred = model(x)
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
            pred = model(X)
            test_loss += loss(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss:{test_loss:.4f}")
        
train_loop(tsdata_train, model, loss_fn, optimizer)
print('train completed.')
test_loop(tsdata_test, model, loss_fn)

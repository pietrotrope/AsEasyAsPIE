import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertConfig
from pruning.tools import retrieve_tuples

DEFAULT_BILSTM = {
    "embed_dim": 300,
    "num_class": 2,
    "batch_size": 256,
    "hidden_dim": 100,
    "dropout_prob": 0.0,
    "num_layers": 2,
    "device": "cpu"
}
DEFAULT_BERT = {
    "n_classes": 2,
    "max_input_length": 256,
    "device": "cpu"
}


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def save_model(self, path):
        tuples = retrieve_tuples(self)

        removed = []
        i = 0
        for layer, name in tuples:
            i += 1
            tmp_name = name.replace("_orig", "").replace("_mask", "")
            if tmp_name != name and tmp_name+str(i) not in removed:
                nn.utils.prune.remove(layer, tmp_name)
                removed.append(tmp_name+str(i))
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class BiLSTMClassifier(Model):
    name = "BiLSTM"

    def __init__(self, params={}):
        super(BiLSTMClassifier, self).__init__()

        if not params:
            tmp = DEFAULT_BILSTM.copy()
            tmp.update(params)
            params = tmp

        self.layers_num = params["num_layers"]
        device = params["device"]

        self.hidden_dim = params["hidden_dim"]

        self.lstm = nn.LSTM(input_size=params["embed_dim"], num_layers=self.layers_num,
                            hidden_size=params["hidden_dim"], bidirectional=True, batch_first=True,
                            device=device)

        self.l1 = nn.Linear(params["hidden_dim"]*2, 256, device=device)
        self.l2 = nn.Linear(256, 128, device=device)
        self.fc = nn.Linear(128, params["num_class"], device=device)
        self.batch_size = params["batch_size"]
        self.use_gpu = False
        if device != "cpu":
            self.use_gpu = True
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, size):
        if self.use_gpu:
            return (Variable(torch.zeros(self.layers_num*2, size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(self.layers_num*2, size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(self.layers_num*2, size, self.hidden_dim)),
                    Variable(torch.zeros(self.layers_num*2, size, self.hidden_dim)))

    def forward(self, text):

        lstm_out, (a, b) = self.lstm(text, self.hidden)
        self.hidden = (a.detach(), b.detach())

        out_forward = lstm_out[range(
            len(lstm_out)), text.shape[1] - 1, :self.hidden_dim]
        out_reverse = lstm_out[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        out = F.relu(self.l1(out_reduced))
        out = F.relu(self.l2(out))
        return self.fc(out)


class BertClassifier(Model):
    name = "BERT"

    def __init__(self, params={}):
        super(BertClassifier, self).__init__()

        if not params:
            tmp = DEFAULT_BERT.copy()
            tmp.update(params)
            params = tmp

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(params["device"])
        self.linear = nn.Linear(
            768, params["n_classes"], device=params["device"])

    def forward(self, input_id, mask, token_type_ids):
        _, pooled_output = self.bert(
            input_id, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return self.linear(pooled_output)

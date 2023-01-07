from torch import nn
import torch


class RNN(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

   def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

   def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class RNN_(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
        super(RNN_, self).__init__()

        self.hidden_size = input_size

        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

   def forward(self, input, hidden):
        hidden = self.W(input) + self.U(hidden)
        hidden = self.tanh(hidden)
        output = self.V(hidden)
        output = self.softmax(output)
        return output, hidden

   def initHidden(self):
        return torch.zeros(1, self.input_size)

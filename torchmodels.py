import torch
import torch.nn as nn
import torch.nn.functional as F

class Lstmcell(nn.Module):
    """This model is LSTM based with two layers
            init:
                args:
                    none
                    
                kwargs:
                    * device <torch.device> : Device for the model to live on. (default: torch.device('cuda:0')) 
                    * hidden_size <int>     : Hidden space between the layers. (default: 51)        
            call:
                args:
                    * stimulus <torch.Tensor> : Tensor containing the input of size (N,I) where:
                                                    - N is the batch size.
                                                    - I is the input size.
                kwargs:
                    none    
            output:
                * tensor <torch.Tensor> : Tensor containing the output of the network of size (N, O) where:
                                                - N is the batch size.
                                                - O is the output size.
    """
    
    def __init__(self, device=torch.device("cuda:0"), hidden_size=51):
        super(Lstmcell, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(1, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, 1)

    def init_weights(self, batchSize):
        h_lstm1 = torch.zeros(batchSize, self.hidden_size).to(self.device)
        c_lstm1 = torch.zeros(batchSize, self.hidden_size).to(self.device)
        h_lstm2 = torch.zeros(batchSize, 1).to(self.device)
        c_lstm2 = torch.zeros(batchSize, 1).to(self.device)
        return h_lstm1, c_lstm1, h_lstm2, c_lstm2

    def forward(self, stimulus):
        batchSize = stimulus.shape[0]
        inputSize = stimulus.shape[1]
        h_1, c_1, h_2, c_2 = self.init_weights(batchSize)
        output = torch.empty((batchSize, inputSize))
        for i in range(inputSize):
            h_1, c_1 = self.lstm1(stimulus[:, i], (h_1, c_1))
            h_2, c_2 = self.lstm2(h_1, (h_2, c_2))
            output[:, i] = h_2.reshape([batchSize])
        return output
    
    def description(self):
        return """\
This model is LSTM based with two layers:
    * hidden space : {}
    * device       : {}\
        """.format(self.hidden_size, self.device)

class Bati(nn.Module):
    """This model is based on the paper by Bati, it has a linear layer and then two lstm layers and a
        layer to adjust the size of the inputs.
        
            init:
                args:
                    none
                    
                kwargs:
                    * device <torch.device> : Device for the model to live on. (default: torch.device('cuda:0')) 
                    * hidden_size <int>     : Hidden space between the layers. (default: 51)
                    * lw <int>              : Width of the input image (default: 7)
                    * lh <int>              : Height of the input image (default: 7)
            call:
                args:
                    * stimulus <torch.Tensor> : Tensor containing the input image (N,I,W,H) where:
                                                    - N is the batch size.
                                                    - I is the sequence size.
                                                    - W is the width size.
                                                    - H is the height size.
                kwargs:
                    none    
            output:
                * tensor <torch.Tensor> : Tensor containing the output of the network of size (N, O) where:
                                                - N is the batch size.
                                                - O is the output size.
    """
    def __init__(self, 
                 device=torch.device("cuda:0"), 
                 hidden_size=51,
                 lw=7, lh=7):
        
        super(Bati, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lw = lw
        self.lh = lh
        self.linear_layer = nn.Linear(self.lw*self.lh, 1)
        self.first_lstm = nn.LSTMCell(1, self.hidden_size)
        self.second_lstm = nn.LSTMCell(self.hidden_size, 1)
        self.adjust_layer = nn.Linear(1, 1)

    def init_weights(self, batch_size):
        h_lstm1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_lstm1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        h_lstm2 = torch.zeros(batch_size, 1).to(self.device)
        c_lstm2 = torch.zeros(batch_size, 1).to(self.device)
        return h_lstm1, c_lstm1, h_lstm2, c_lstm2

    def forward(self, stimulus):
        batch_size = stimulus.shape[0]
        h_1, c_1, h_2, c_2 = self.init_weights(batch_size)
        output = torch.empty(torch.Size([stimulus.shape[0], stimulus.shape[1]]))
        for i in range(stimulus.shape[1]):
            unravel = stimulus[:, i, :, :].reshape(batch_size, -1).to(self.device)
            out_l = F.relu(self.linear_layer(unravel))
            h_1, c_1 = self.first_lstm(out_l, (h_1, c_1))
            h_2, c_2 = self.second_lstm(h_1, (h_2, c_2))
            out = self.adjust_layer(h_2)
            output[:, i] = out.reshape([batch_size])
        return output
    
    def description(self):
        return """\
This model is based on the paper by Bati, it has a linear layer and then two lstm layers and a \
layer to adjust the size of the inputs.
    * hei x weig   : {}x{}
    * hidden space : {}
    * device       : {}\
        """.format(self.lh,
                   self.lw,
                   wself.hidden_size, 
                   self.device)
    
items = [Lstmcell, Bati]

def usage(verbose=True):
    for item in items:
        print(item.__name__, ":")
        if verbose:
            print(item.__doc__, "\n")


if '__name__' == '__main__':
    usage()
    
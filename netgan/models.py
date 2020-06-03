import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):

    def __init__(self, H_inputs, H, z_dim, N, rw_len, temp):
        '''
            H_inputs: input dimension
            H:        hidden dimension
            z_dim:    latent dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
            temp:     temperature for the gumbel softmax
        '''
        super(Generator, self).__init__()
        self.intermediate = nn.Linear(z_dim, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate.weight)
        torch.nn.init.zeros_(self.intermediate.bias)
        self.c_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up.weight)
        torch.nn.init.zeros_(self.c_up.bias)
        self.h_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up.weight)
        torch.nn.init.zeros_(self.h_up.bias)
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)


        self.W_up = nn.Linear(H, N).type(torch.float64)
        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        self.rw_len = rw_len
        self.temp = temp
        self.H = H
        self.latent_dim = z_dim
        self.N = N
        self.H_inputs = H_inputs

    def forward(self, latent, inputs, device='cuda'):   # h_down = input_zeros
        intermediate = torch.tanh(self.intermediate(latent))
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        out = []  # gumbel_noise = uniform noise [0, 1]
        for i in range(self.rw_len):
            hh, cc = self.lstmcell(inputs, hc)
            hc = (hh, cc)
            h_up = self.W_up(hh)                # blow up to dimension N using W_up
            h_sample = self.gumbel_softmax_sample(h_up, self.temp, device)
            inputs = self.W_down(h_sample)      # back to dimension H (in netgan they reduce the dimension to d)
            out.append(h_sample)
        return torch.stack(out, dim=1)

    def sample_latent(self, num_samples, device):
        return torch.randn((num_samples, self.latent_dim)).type(torch.float64).to(device)


    def sample(self, num_samples, device):
        noise = self.sample_latent(num_samples, device)
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(device)
        generated_data = self(noise,  input_zeros, device)
        return generated_data

    def sample_discrete(self, num_samples, device):
        with torch.no_grad():
            proba = self.sample(num_samples, device)
        return np.argmax(proba.cpu().numpy(), axis=2)

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand(logits.shape, dtype=torch.float64)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits,  temperature, device, hard=True):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(device)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=1)
        if hard:
            y_hard = torch.max(y, 1, keepdim=True)[0].eq(y).type(torch.float64).to(device)
            y = (y_hard - y).detach() + y
        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)

    #def reset_weights(self):
    #    import h5py
    #    weights = h5py.File(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\weights.h5', 'r')
    #    self.intermediate.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('intermediate')).T).type(torch.float64))
    #    self.intermediate.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('intermediate_bias'))).type(torch.float64))
    #    self.c_up.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('c')).T).type(torch.float64))
    #    self.c_up.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('c_bias'))).type(torch.float64))
    #    self.h_up.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('h')).T).type(torch.float64))
    #    self.h_up.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('h_bias'))).type(torch.float64))
    #    self.lstmcell.cell.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('generator_lstm')).T).type(torch.float64))
    #    self.lstmcell.cell.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('generator_lstm_bias'))).type(torch.float64))
    #    self.W_up.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('W_up_generator')).T).type(torch.float64))
    #    self.W_up.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('W_up_generator_bias'))).type(torch.float64))
    #    self.W_down.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('W_down_generator')).T).type(torch.float64))


class Discriminator(nn.Module):
    def __init__(self, H_inputs, H, N, rw_len):
        '''
            H_inputs: input dimension
            H:        hidden dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
        '''
        super(Discriminator, self).__init__()
        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)
        self.lin_out = nn.Linear(H, 1, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        torch.nn.init.zeros_(self.lin_out.bias)
        self.H = H
        self.N = N
        self.rw_len = rw_len
        self.H_inputs = H_inputs

    def forward(self, x):
        x = x.view(-1, self.N)
        xa = self.W_down(x)
        xa = xa.view(-1, self.rw_len, self.H_inputs)
        hc = self.init_hidden(xa.size(0))
        for i in range(self.rw_len):
            hc = self.lstmcell(xa[:, i, :], hc)
        out = hc[0]
        pred = self.lin_out(out)
        return pred

    def init_inputs(self, num_samples):
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        weight = next(self.parameters()).data
        return (weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64), weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64))

    #def reset_weights(self):
    #    import h5py
    #    weights = h5py.File(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\weights.h5', 'r')
    #    self.W_down.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('W_down_discriminator')).T).type(torch.float64))
    #    self.lin_out.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('discriminator_out')).T).type(torch.float64))
    #    self.lin_out.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('discriminator_out_bias'))).type(torch.float64))
    #    self.lstmcell.cell.weight = torch.nn.Parameter(torch.tensor(np.array(weights.get('discriminator_lstm')).T).type(torch.float64))
    #    self.lstmcell.cell.bias = torch.nn.Parameter(torch.tensor(np.array(weights.get('discriminator_lstm_bias'))).type(torch.float64))

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.Linear(input_size+hidden_size, 4 * hidden_size, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = torch.cat((x, hx), dim=1)
        gates = self.cell(gates)

        ingate, cellgate, forgetgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(torch.add(forgetgate, 1.0))
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)
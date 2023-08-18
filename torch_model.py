import numpy as np

import torch    
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros

### LOSS FUNCTIONS #####################################################################
class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__();

    def forward(self, obs, prd):
        err_u = torch.abs(obs[:, 0, :, :]-prd[:, 0, :, :])
        err_v = torch.abs(obs[:, 1, :, :]-prd[:, 1, :, :])
        err_sic = torch.abs(obs[:, 2, :, :]-prd[:, 2, :, :])
        err_sit = torch.abs(obs[:, 3, :, :]-prd[:, 3, :, :])
        
        err_sum = torch.mean((err_u + err_v) + err_sic + err_sit)
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum*100    
    
class physics_loss(nn.Module):
    def __init__(self):
        super(physics_loss, self).__init__();

    def forward(self, obs, prd):
        err_u = torch.abs(obs[:, 0, 1:-1, 1:-1]-prd[:, 0, 1:-1, 1:-1])
        err_v = torch.abs(obs[:, 1, 1:-1, 1:-1]-prd[:, 1, 1:-1, 1:-1])
        err_sic = torch.abs(obs[:, 2, 1:-1, 1:-1]-prd[:, 2, 1:-1, 1:-1])
        
        err_sum = torch.mean((err_u + err_v) + err_sic)
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        
        u = prd[:, 0, :, :]
        v = prd[:, 1, :, :]
        d_sic = prd[:, 2, 1:-1, 1:-1]
        
        dy = v[:, 2:, 1:-1] - v[:, :-2, 1:-1]      
        dx = u[:, 1:-1, 2:] - u[:, 1:-1, :-2]
         
        div = dx + dy
        
        # SIC change
        err_phy = torch.mean(torch.where((div > 0.03) & (d_sic > 0.1), div * d_sic, 0))
        # err_phy = torch.mean(torch.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        
        w = torch.tensor(10.0)
        err_sum += w*err_phy
        
        return err_sum*100
    
### MAKE INPUT DATASETS #########################################################
def convert_cnn_input2D(data_input, data_output, seq_days, months, years, dayint = 7):
    # Input & output should be entire images for CNN
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)

    cnn_input = np.zeros([n_samples-dayint, row, col, var_ip * dayint], dtype = np.float16)
    cnn_output = np.zeros([n_samples-dayint, row, col, var_op * dayint], dtype = np.float16)
    valid = []
    
    for n in range(0, n_samples-dayint):
        if seq_days[n] + dayint == seq_days[n+dayint]:
            valid.append(n)
            for i in range(0, dayint):
                for v in range(0, var_ip):            
                    cnn_input[n, :, :, v+i*var_ip] = (data_input[n+i, :, :, v])
                if v in range(0, var_op):
                    cnn_output[n, :, :, v+i*var_op] = (data_output[n+i, :, :, v])
                
    return cnn_input[valid, :, :, :], cnn_output[valid, :, :, :], seq_days[valid], months[valid], years[valid]

### ML MODELS #####################################################################
# CNN model
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_filters=32, kernel = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv5 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv6 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv7 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv8 = nn.Conv2d(n_filters, n_outputs, kernel, padding = "same")

    def forward(self, x):
        # x = F.tanh(self.conv1(x)) #F.leaky_relu(self.conv1(x))
        # x = F.tanh(self.conv2(x)) #F.leaky_relu(self.conv2(x))
        # x = F.tanh(self.conv3(x)) #F.leaky_relu(self.conv3(x))
        # x = F.tanh(self.conv4(x)) #F.leaky_relu(self.conv4(x))
        # x = F.tanh(self.conv5(x)) #F.leaky_relu(self.conv5(x))
        # x = F.tanh(self.conv6(x)) #F.leaky_relu(self.conv6(x))
        # x = F.tanh(self.conv7(x)) #F.leaky_relu(self.conv7(x))
        # x = F.tanh(self.conv8(x)) #F.leaky_relu(self.conv8(x))
        
        x = F.leaky_relu(self.conv1(x), negative_slope=1)
        x = F.leaky_relu(self.conv2(x), negative_slope=1)
        x = F.leaky_relu(self.conv3(x), negative_slope=1)
        x = F.leaky_relu(self.conv4(x), negative_slope=1)
        x = F.leaky_relu(self.conv5(x), negative_slope=1)
        x = F.leaky_relu(self.conv6(x), negative_slope=1)
        x = F.leaky_relu(self.conv7(x), negative_slope=1)
        x = F.leaky_relu(self.conv8(x), negative_slope=1)
        
        # x = F.linear(self.conv1(x), weight = 1)
        # x = F.linear(self.conv2(x), weight = 1)
        # x = F.linear(self.conv3(x), weight = 1)
        # x = F.linear(self.conv4(x), weight = 1)
        # x = F.linear(self.conv5(x), weight = 1)
        # x = F.linear(self.conv6(x), weight = 1)
        # x = F.linear(self.conv7(x), weight = 1)
        # x = F.linear(self.conv8(x), weight = 1)
        
        return x
    
class GCNet(torch.nn.Module):
    def __init__(self, hidden_channels = 32):
        super().__init__()
        # torch.manual_seed(1234567)
        self.conv1 = GCNConv(8, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, 3)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index), negative_slope=1); #self.conv1(x)
        x = F.leaky_relu(self.conv2(x, edge_index), negative_slope=1);
        x = F.leaky_relu(self.conv3(x, edge_index), negative_slope=1);
        x = F.leaky_relu(self.conv4(x, edge_index), negative_slope=1);
        x = F.leaky_relu(self.conv5(x, edge_index), negative_slope=1);
        # x = x.relu() #x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        return x

# Convolutional LSTM cell
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs

# UNET model
class UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
         
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # input: 320x320x3
        self.e11 = nn.Conv2d(n_inputs, 64, kernel_size=3, padding="same") # output: 320x320x32
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding="same") # output: 320x320x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x32

        # input: 160x160x32
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding="same") # output: 160x160x64
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding="same") # output: 160x160x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x80x64

        # input: 80x80x64
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding="same") # output: 80x80x128
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding="same") # output: 80x80x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x40x128

        # input: 40x40x128
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding="same") # output: 40x40x256
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding="same") # output: 40x40x256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 20x20x256

        # input: 20x20x256
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding="same") # output: 20x20x512
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding="same") # output: 20x20x512

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding="same")
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding="same")

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding="same")
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding="same")

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding="same")

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding="same")
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding="same")

        # Output layer
        self.outconv = nn.Conv2d(64, n_outputs, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        xe11 = F.tanh(self.e11(x))
        xe12 = F.tanh(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.tanh(self.e21(xp1))
        xe22 = F.tanh(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.tanh(self.e31(xp2))
        xe32 = F.tanh(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.tanh(self.e41(xp3))
        xe42 = F.tanh(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.tanh(self.e51(xp4))
        xe52 = F.tanh(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.tanh(self.d11(xu11))
        xd12 = F.tanh(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.tanh(self.d21(xu22))
        xd22 = F.tanh(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.tanh(self.d31(xu33))
        xd32 = F.tanh(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.tanh(self.d41(xu44))
        xd42 = F.tanh(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

# Graph convolutional LSTM
class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C
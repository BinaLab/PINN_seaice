import numpy as np

import torch    
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros

### LOSS FUNCTIONS #####################################################################
class vel_loss(nn.Module):
    def __init__(self):
        super(vel_loss, self).__init__();

    def forward(self, obs, prd):
        u_o = obs[:, 0, :, :]; v_o = obs[:, 1, :, :]
        u_p = prd[:, 0, :, :]; v_p = prd[:, 1, :, :]
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        # theta = (u_o*u_p+v_o*v_p)/(vel_o*vel_p)
        # theta = (1 - theta**2)**0.5
        # theta = torch.where(theta >= 0, theta, 0)
        # err_theta = torch.abs(theta)

        err_u = torch.abs(u_o - u_p)
        err_v = torch.abs(v_o - v_p)
        err_vel = torch.abs(vel_o - vel_p)        

        err_sum = torch.mean((err_u + err_v + err_vel))*100
        # err_sum += torch.nanmean(err_theta)/10000
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum  
    
class single_loss(nn.Module):
    def __init__(self, landmask):
        super(single_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd):
        n_outputs = obs.size()[1]
        err_sum = 0
        for i in range(0, n_outputs):
            err = torch.square(obs[:, i, :, :] - prd[:, i, :, :])
            # err = torch.mean(err, dim=0)[self.landmask == 0]
            err_sum += torch.mean(err)**0.5*100
        return err_sum

class custom_loss(nn.Module):
    def __init__(self, landmask):
        super(custom_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd):
        sic = prd[:, 2, :, :]*100
        u_o = obs[:, 0, :, :]*30; v_o = obs[:, 1, :, :]*30
        u_p = prd[:, 0, :, :]*30; v_p = prd[:, 1, :, :]*30
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        theta = torch.acos((u_o*u_p+v_o*v_p)/(vel_o*vel_p))
        theta = torch.where(torch.isnan(theta), 0, theta)

        err_u = torch.square(u_o - u_p) #[sic > 0]
        err_v = torch.square(v_o - v_p) #[sic > 0]
        err_vel = torch.square(vel_o - vel_p) #[sic > 0]
        err_theta = torch.abs(theta)
        
        err1 = torch.mean(err_u + err_v, dim=0)[torch.where(self.landmask == 0)]
        err_sum = torch.mean(err1)*1000 

        err_sic = torch.square(obs[:, 2, :, :]-prd[:, 2, :, :])
        
        neg_sic = torch.where(prd[:, 2, :, :] < 0, abs(prd[:, 2, :, :]), 0)
        err2 = torch.mean(err_sic, dim=0)[torch.where(self.landmask == 0)]
        err_sum += torch.mean(err2)*1000
        
        if obs.size()[1] > 3:
            err_sit = torch.abs(obs[:, 3, :, :]-prd[:, 3, :, :])  
            neg_sit = torch.where(prd[:, 3, :, :] < 0, abs(prd[:, 3, :, :]), 0)
            err3 = torch.mean(err_sit, dim=0)[torch.where(self.landmask == 0)]   
            err_sum += torch.mean(err3)*5000
        
        # err_sum += torch.mean(err_sic + err_sit)*100
        # err_sum += torch.nanmean(err_theta)*0.5/3.141592
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum   

def corrcoef(x, y):
    x = x.flatten()
    y = y.flatten()
    xm = torch.mean(x)
    ym = torch.mean(y)

    r1 = torch.sum((x-xm)*(y-ym))
    r2 = torch.sum(torch.square(x-xm))*torch.sum(torch.square(y-ym))
    r = r1/(r2**0.5)
    return r
    
class physics_loss(nn.Module):
    def __init__(self, landmask):
        super(physics_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd, sic0):
        
        sic_th = 0.001
        
        sic_p = prd[:, 2, :, :]
        # sic_p[sic_p > 1] = 1
        # sic_p[sic_p < 0] = 0
        sic_o = obs[:, 2, :, :]
        u_o = obs[:, 0, :, :]*50; v_o = obs[:, 1, :, :]*50
        u_p = prd[:, 0, :, :]*50; v_p = prd[:, 1, :, :]*50
        
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        # u_p[sic_p <= sic_th] = 0
        # v_p[sic_p <= sic_th] = 0
        
        err_u = torch.square(u_o - u_p) #[sic > 0]
        err_v = torch.square(v_o - v_p) #[sic > 0]
        
        sicmask = torch.max(sic_o, dim=0)[0]
        err1 = torch.mean(err_u + err_v, dim=0)[torch.where(self.landmask == 0)]
        err_sum = torch.mean(err1)

        err_sic = torch.square(sic_o - sic_p)
        
        err2 = torch.mean(err_sic, dim=0)[torch.where(self.landmask == 0)]
        err_sum += torch.mean(err2)*2500
        
        # if obs.size()[1] > 3:
        #     sit_p = prd[:, 3, :, :]
        #     sit_o = obs[:, 3, :, :]
        #     err_sit = torch.square(sit_o - sit_p)
        #     neg_sit = torch.where(sit_p < 0, abs(sit_p), 0)
        #     err3 = torch.mean(err_sit + neg_sit, dim=0)[torch.where(self.landmask == 0)]   
        #     err_sum += torch.mean(err3)*50
        
        # physics loss ===============================================
        ## Where SIC < 0 ==> sea ice drift = 0!
        err_phy = 0
        
        ## Negative or positive SIC
        neg_sic = torch.where(sic_p < 0, err_sic, 0)
        pos_sic = torch.where(sic_p > 1, err_sic, 0)     
        err5 = torch.mean(neg_sic + pos_sic, dim=0)[torch.where(self.landmask == 0)]
        err_phy += torch.mean(err5)
        
        # advection
        advc = sic_p*0
        dx = (sic_p[:, 1:-1, 2:]-sic_p[:, 1:-1, :-2]) + (sic_p[:, 2:, 2:]-sic_p[:, 2:, :-2]) + (sic_p[:, :-2, 2:]-sic_p[:, :-2, :-2])
        dy = (sic_p[:, 2:, 1:-1]-sic_p[:, :-2, 1:-1]) + (sic_p[:, 2:, 2:]-sic_p[:, :-2, 2:]) + (sic_p[:, 2:, :-2]-sic_p[:, :-2, :-2])    
        advc[:, 1:-1, 1:-1] = (u_p[:, 1:-1, 1:-1]*dx/3 + v_p[:, 1:-1, 1:-1]*dy/3)/25
        
        # divergence
        divc = sic_p*0
        dx = (u_p[:, 1:-1, 2:]-u_p[:, 1:-1, :-2]) + (u_p[:, 2:, 2:]-u_p[:, 2:, :-2]) + (u_p[:, :-2, 2:]-u_p[:, :-2, :-2])
        dy = (v_p[:, 1:-1, 2:]-v_p[:, 1:-1, :-2]) + (v_p[:, 2:, 2:]-v_p[:, 2:, :-2]) + (v_p[:, :-2, 2:]-v_p[:, :-2, :-2])
        divc[:, 1:-1, 1:-1] = dx/3 + dy/3
        divc = divc*sic_p/25
        
        dsic = sic_p - sic0
        
        residual = dsic + advc
        
        # SIC change
        err_res = torch.mean(torch.where(abs(residual) > 1, abs(residual)-1, 0), dim = 0)[torch.where(self.landmask == 0)]
        err_phy += torch.mean(err_res)
        
        r = corrcoef(dsic, advc)
        if r > 0:
            err_phy += r
        # err_phy = torch.mean(torch.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        
        w = torch.tensor(10.0)
        err_sum += w*err_phy
        
        return err_sum    
    
    
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, obs, sid, sic, sit):
        
        # Sea ice drift error
        err_u = torch.abs(obs[:, 0, :, :] - sid[:, 0, :, :])
        err_v = torch.abs(obs[:, 1, :, :] - sid[:, 1, :, :])
        loss0 = torch.mean((err_u + err_v))*100 
        
        # SIC error
        err_sic = torch.abs(obs[:, 2, :, :]-sic)
        loss1 = torch.mean(err_sic)*100
        
        # SIT error
        err_sit = torch.abs(obs[:, 3, :, :]-sit)
        loss2 = torch.mean(err_sit)*100

#         precision0 = torch.exp(-self.log_vars[0])
#         loss0 = precision0*loss0 + self.log_vars[0]

#         precision1 = torch.exp(-self.log_vars[1])
#         loss1 = precision1*loss1 + self.log_vars[1]

#         precision2 = torch.exp(-self.log_vars[2])
#         loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2
    
### MAKE INPUT DATASETS #########################################################
def convert_cnn_input2D(data_input, data_output, days, months, years, dayint = 3, forecast = 3, exact = False):
    # dayint: days before forecast (use as input features)
    # forecast: lead day for forecasting (output features)
    # exact: if True, only that exact date is forecasted; if False, all days before the lead day is forecasted
    # Input & output should be entire images for CNN
    
    # Cehck sequential days
    seq_days = []
    step = 0

    for i in range(0, len(days)):
        if (days[i] ==1) & (years[i] != years[0]):
            step += days[i-1]
        seq_days.append(days[i] + step)

    seq_days = np.array(seq_days)
    
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)

    cnn_input = np.zeros([n_samples, row, col, var_ip * dayint], dtype = np.float16)
    if exact:
        cnn_output = np.zeros([n_samples, row, col, var_op], dtype = np.float16)
    else:
        cnn_output = np.zeros([n_samples, row, col, var_op * forecast], dtype = np.float16)
    valid = []
    
    for n in range(dayint-1, n_samples-forecast):
        if seq_days[n+forecast] - seq_days[n-dayint+1] == dayint + forecast-1:
            valid.append(n)
            for i in range(0, dayint):
                for v in range(0, var_ip):            
                    cnn_input[n, :, :, v+i*var_ip] = (data_input[n-i, :, :, v]).astype(np.float16)
            # if v in range(0, var_op):
            if exact:
                cnn_output[n, :, :, :] = (data_output[n+forecast-1, :, :, :]).astype(np.float16)
            else:
                for j in range(0, forecast):
                    for v in range(0, var_op):            
                        cnn_output[n, :, :, v+j*var_op] = (data_output[n+j, :, :, v]).astype(np.float16)
                
                
    return cnn_input[valid, :, :, :], cnn_output[valid, :, :, :], days[valid], months[valid], years[valid]

### ML MODELS #####################################################################
class FC(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(n_inputs, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        return x

class linear_regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, row, col):
        super(linear_regression, self).__init__()        
        self.asiu = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsiu = torch.nn.Parameter(torch.ones(1, row, col)*0.5)
        self.asiv = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsiv = torch.nn.Parameter(torch.ones(1, row, col)*0.5)
        self.asic = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsic = torch.nn.Parameter(torch.ones(1, row, col)*0.5)

    def forward(self, x):
        siu = torch.sum(self.asiu*x, dim=1) + self.bsiu
        siv = torch.sum(self.asiv*x, dim=1) + self.bsiv
        sic = torch.sum(self.asic*x, dim=1) + self.bsic
        out = torch.cat([siu.unsqueeze(1), siv.unsqueeze(1), sic.unsqueeze(1)], dim=1)
        return out
    
# class Linear_regression(nn.Module):
#     def __init__(self, n_inputs, n_outputs):
#         super().__init__()
#         self.activation = nn.Tanh()
#         self.fc1 = nn.Linear(n_inputs, 128)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, n_outputs)

#     def forward(self, x):
        
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
        
#         return x

# CNN model
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_filters=32, kernel = 5):
        super().__init__()
        self.activation = nn.Tanh()
        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv5 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv6 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv7 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv8 = nn.Conv2d(n_filters, n_outputs, kernel, padding = "same")

    def forward(self, x):
        
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7(x))
        x = self.activation(self.conv8(x))
        
        return x
    
class CNN_flatten(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, extent, n_filters=32, kernel = 5):
        super().__init__()
        self.activation = nn.Tanh()
        self.landmask = landmask
        self.n_outputs = n_outputs
        self.extent = extent
        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 128*128
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 64*64
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 32*32
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 16*16
        self.conv5 = nn.Conv2d(n_filters, 4, kernel, padding = "same")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 8*8
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=int(4*(extent/(2**5))**2), out_features=n_outputs* extent * extent)
        # self.fc2 = nn.Linear(in_features=4*10 * 10, out_features=4 * 80 * 80)
        # self.upconv1 = nn.ConvTranspose2d(4, n_filters, kernel_size=2, stride=2) # 160*160
        # self.upconv2 = nn.ConvTranspose2d(n_filters, n_outputs, kernel_size=2, stride=2) # 320*320
        # self.fc2 = nn.Linear(in_features=10, out_features=n_outputs*320*320)

    def forward(self, x):
        # x = F.tanh(self.conv1(x)) #F.leaky_relu(self.conv1(x))
        # x = F.tanh(self.conv2(x)) #F.leaky_relu(self.conv2(x))
        # x = F.tanh(self.conv3(x)) #F.leaky_relu(self.conv3(x))
        # x = F.tanh(self.conv4(x)) #F.leaky_relu(self.conv4(x))
        # x = F.tanh(self.conv5(x)) #F.leaky_relu(self.conv5(x))
        # x = F.tanh(self.conv6(x)) #F.leaky_relu(self.conv6(x))
        # x = F.tanh(self.conv7(x)) #F.leaky_relu(self.conv7(x))
        # x = F.tanh(self.conv8(x)) #F.leaky_relu(self.conv8(x))
        
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        x = self.activation(self.conv4(x))
        x = self.pool4(x)
        x = self.activation(self.conv5(x))
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = x.reshape(-1, self.n_outputs, self.extent, self.extent)
        x = x * (self.landmask == 0)
        # x = F.leaky_relu(self.upconv1(x), negative_slope=0.1)
        # x = F.leaky_relu(self.upconv2(x), negative_slope=0.1)
        
        return x



class CNN_flatten_hydra(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_filters=64, kernel = 5):
        super().__init__()

        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")     
        self.bn1 = nn.BatchNorm2d(n_filters)      
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 160*160
        
        self.activation = nn.LeakyReLU(negative_slope=1)
        
        self.conv2_1 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn2_1 = nn.BatchNorm2d(n_filters)
        # self.activation = nn.Tanh()
        self.conv2_2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn2_2 = nn.BatchNorm2d(n_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 80*80
        
        self.conv3_1 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn3_1 = nn.BatchNorm2d(n_filters)
        # self.activation = nn.Tanh()   
        self.conv3_2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn3_2 = nn.BatchNorm2d(n_filters)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 40*40
        
        self.conv4_1 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn4_1 = nn.BatchNorm2d(n_filters)
        # self.activation = nn.Tanh()   
        self.conv4_2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.bn4_2 = nn.BatchNorm2d(n_filters)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 20*20
        
        self.conv5_1 = nn.Conv2d(n_filters, 16, kernel, padding = "same")
        self.bn5_1 = nn.BatchNorm2d(16)
        # self.activation = nn.Tanh()   
        self.conv5_2 = nn.Conv2d(16, 8, kernel, padding = "same")
        self.bn5_2 = nn.BatchNorm2d(8)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 10*10
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8 * 10 * 10, out_features=2*320 * 320)
        self.fc2 = nn.Linear(in_features=8 * 10 * 10, out_features=1*320 * 320)
        self.fc3 = nn.Linear(in_features=8 * 10 * 10, out_features=1*320 * 320)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.activation(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.activation(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x1 = self.fc1(x)
        sid_head = x1.reshape(-1, 2, 320, 320)
        x2 = self.fc2(x)
        sic_head = x2.reshape(-1, 1, 320, 320)
        x3 = self.fc3(x)
        sit_head = x3.reshape(-1, 1, 320, 320)
        
        out = torch.cat([sid_head, sic_head, sit_head], dim=1)
        
        return out
    
class CNN_hydra(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_filters=64, kernel = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 160*160
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 80*80
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 40*40
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 20*20
        self.conv5 = nn.Conv2d(n_filters, n_filters*2, kernel, padding = "same")
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # size: 10*10
        
        self.conv_uv = nn.Conv2d(n_filters*2, 2, kernel, padding = "same")
        self.conv_sic = nn.Conv2d(n_filters*2, 1, kernel, padding = "same")
        self.conv_sit = nn.Conv2d(n_filters*2 , 1, kernel, padding = "same")
        
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=4 * 10 * 10, out_features=2*320 * 320)
        # self.fc2 = nn.Linear(in_features=4 * 10 * 10, out_features=1*320 * 320)
        # self.fc3 = nn.Linear(in_features=4 * 10 * 10, out_features=1*320 * 320)

    def forward(self, x):
        x = F.tanh(self.conv1(x)) #F.leaky_relu(self.conv1(x))
        x = F.tanh(self.conv2(x)) #F.leaky_relu(self.conv2(x))
        x = F.tanh(self.conv3(x)) #F.leaky_relu(self.conv3(x))
        x = F.tanh(self.conv4(x)) #F.leaky_relu(self.conv4(x))
        x = F.tanh(self.conv5(x)) #F.leaky_relu(self.conv5(x))
        # x = F.tanh(self.conv6(x)) #F.leaky_relu(self.conv6(x))
        # x = F.tanh(self.conv7(x)) #F.leaky_relu(self.conv7(x))
        # x = F.tanh(self.conv8(x)) #F.leaky_relu(self.conv8(x))
        
        # x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # x = self.pool1(x)
        # x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        # x = self.pool2(x)
        # x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        # x = self.pool3(x)
        # x = F.leaky_relu(self.conv4(x), negative_slope=0.1)
        # x = self.pool4(x)
        # x = F.leaky_relu(self.conv5(x), negative_slope=0.1)
        # x = self.pool5(x)
        # x = self.flatten(x)
        # x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        # # x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        # x = x.reshape(-1, 4, 320, 320)
        
        sid_head = F.tanh(self.conv_uv(x))
        sic_head = F.tanh(self.conv_sic(x))
        sit_head = F.tanh(self.conv_sit(x))
        
        # x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        # sid_head = x1.reshape(-1, 2, 320, 320)
        # x2 = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        # sic_head = x2.reshape(-1, 1, 320, 320)
        # x3 = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        # sit_head = x3.reshape(-1, 1, 320, 320)
        
        out = torch.cat([sid_head, sic_head, sit_head], dim=1)
        
        return out
    
    
class GCNet(torch.nn.Module):
    def __init__(self, ch_input, ch_output, hidden_channels = 32):
        super().__init__()
        # torch.manual_seed(1234567)
        self.conv1 = GCNConv(ch_input, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, ch_output)

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

# Early branch UNET model
class UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, k=3):
        super().__init__()
        
        self.activation = nn.Tanh()
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        # input: 320x320x64
        self.siu_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siu_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siu_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siu_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siu_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siu_dc1 = decoder(512, 256) # output: 80x80x256
        self.siu_dc2 = decoder(256, 128) # output: 160x160x128
        self.siu_dc3 = decoder(128, 64) # output: 320x320x64     

        # Output layer
        self.out_conv = nn.Conv2d(64, n_outputs, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)        
        
        ##### Encoder 1 #####
        xe1_siu, xe1b_siu = self.siu_ec1(x) # SIU
        
        ##### Encoder 2 #####
        xe2_siu, xe2b_siu = self.siu_ec2(xe1_siu) # SIU

        ##### Encoder 3 #####
        xe3_siu, xe3b_siu = self.siu_ec3(xe2_siu) # SIU
        
        ##### Bottom bridge #####
        # SID
        xe41_siu = self.activation(self.siu_ec41(xe3_siu))
        xe42_siu = self.activation(self.siu_ec42(xe41_siu))
        # output: 40x40x512
        
        ##### Decoder 1 #####
        # SIU
        xd1_siu = self.siu_dc1(xe42_siu, xe3b_siu)
        # Weighting block 5
        
        ##### Decoder 2 #####
        # SIU
        xd2_siu = self.siu_dc2(xd1_siu, xe2b_siu)
        # Weighting block 6        
        
        ##### Decoder 3 #####
        # SIU
        xd3_siu = self.siu_dc3(xd2_siu, xe1b_siu)
        out = self.out_conv(xd3_siu)

        out = out * (self.landmask == 0)

        return out

    
class TCL_block(nn.Module):
    def __init__(self, ch, row, col, k=3, w=0.5):
        super(TCL_block,self).__init__()
        self.activation = nn.Tanh()
        self.a11 = torch.nn.Parameter(torch.ones(1, ch, row, col)*w)
        self.a12 = torch.nn.Parameter(torch.ones(1, ch, row, col)*w)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64
        self.a21 = torch.nn.Parameter(torch.ones(1, ch, row, col)*w)
        self.a22 = torch.nn.Parameter(torch.ones(1, ch, row, col)*w)

    def forward(self, x1, x2):
        x = self.a11.repeat(x1.size()[0], 1, 1, 1)*x1 + self.a12.repeat(x2.size()[0], 1, 1, 1)*x2
        x = self.activation(self.conv2(self.conv1(x)))
        x1 = self.a21.repeat(x1.size()[0], 1, 1, 1)*x
        x2 = self.a22.repeat(x2.size()[0], 1, 1, 1)*x
        return x1, x2    

# Attention blocks
class AttBlock(nn.Module):
    def __init__(self, ch, row, col, k=1, w=0.5):
        super(AttBlock,self).__init__()
        self.activation = nn.Tanh()
        self.a11 = torch.nn.Parameter(torch.ones(ch, row, col)*w)
        self.a12 = torch.nn.Parameter(torch.ones(ch, row, col)*w)
        self.conv0 = nn.Conv2d(ch, ch, kernel_size=1, padding="same")
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64

    def forward(self, x1, x2):
        x = x1*self.a11 + x2*self.a12
        x = self.activation(self.conv0(x))
        x1 = x1 + self.conv1(x)
        x2 = x2 + self.conv2(x)
        return x1, x2
    
# Weighting blocks
class WB(nn.Module):
    def __init__(self, ch, row, col, k=3, w=0.5):
        super(WB,self).__init__()
        self.activation = nn.Tanh()
        self.a11 = torch.nn.Parameter(torch.ones(row, col)*w)
        self.a12 = torch.nn.Parameter(torch.ones(row, col)*w)
        self.a13 = torch.nn.Parameter(torch.ones(row, col)*w)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=k, padding="same") # output: 160x160x64
        self.a21 = torch.nn.Parameter(torch.ones(row, col)*w)
        self.a22 = torch.nn.Parameter(torch.ones(row, col)*w)
        self.a23 = torch.nn.Parameter(torch.ones(row, col)*w)

    def forward(self, x1, x2, x3):
        x = x1*self.a11 + x2*self.a12 + x3*self.a13
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x1 = x*self.a21
        x2 = x*self.a22
        x3 = x*self.a23
        return x1, x2, x3  
    
class encoder(nn.Module):
    def __init__(self, ch1, ch2, k=3):
        super(encoder,self).__init__()
        self.activation = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)
        self.e11 = nn.Conv2d(ch1, ch2, kernel_size=k, padding="same") # output: 320x320x64
        self.e12 = nn.Conv2d(ch2, ch2, kernel_size=k, padding="same") # output: 320x320x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x64

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.e11(x))
        xb = self.activation(self.e12(x))
        x = self.pool1(xb)
        return x, xb
    
class decoder(nn.Module):
    def __init__(self, ch1, ch2, k=3):
        super(decoder,self).__init__()
        self.activation = nn.Tanh() #nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.upconv1 = nn.ConvTranspose2d(ch1, ch2, kernel_size=2, stride=2) # output: 80x80x256
        self.d11 = nn.Conv2d(ch1, ch2, kernel_size=k, padding="same") # output: 80x80x256
        self.d12 = nn.Conv2d(ch2, ch2, kernel_size=k, padding="same") # output: 80x80x256

    def forward(self, x, x0):        
        x = self.upconv1(x)        
        x = torch.cat([x, x0], dim=1) 
        x = self.dropout(x)
        x = self.activation(self.d11(x))
        x = self.activation(self.d12(x))
        return x
    
# Triple-sharing UNET model
class TS_UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, extent, k=3):
        super().__init__()
        
        self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        ##### SIU BRANCH #####
        # input: 320x320x64
        self.siu_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siu_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siu_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siu_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siu_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siu_dc1 = decoder(512, 256) # output: 80x80x256
        self.siu_dc2 = decoder(256, 128) # output: 160x160x128
        self.siu_dc3 = decoder(128, 64) # output: 320x320x64     
        
        ##### SIV BRANCH #####
        # input: 320x320x64
        self.siv_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siv_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siv_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siv_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siv_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siv_dc1 = decoder(512, 256) # output: 80x80x256
        self.siv_dc2 = decoder(256, 128) # output: 160x160x128
        self.siv_dc3 = decoder(128, 64) # output: 320x320x64 
        
        ##### SIC BRANCH #####
        # input: 320x320x64
        self.sic_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.sic_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.sic_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.sic_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.sic_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.sic_dc1 = decoder(512, 256) # output: 80x80x256
        self.sic_dc2 = decoder(256, 128) # output: 160x160x128
        self.sic_dc3 = decoder(128, 64) # output: 320x320x64 
        
        
        ##### Weighting Blocks #####
        self.wb1 = WB(64, int(extent/2), int(extent/2), k, 0)        
        self.wb2 = WB(128, int(extent/4), int(extent/4), k, 0)
        self.wb3 = WB(256, int(extent/8), int(extent/8), k, 0)
        self.wb4 = WB(512, int(extent/8), int(extent/8), k, 0)
        self.wb5 = WB(256, int(extent/4), int(extent/4), k, 0)
        self.wb6 = WB(128, int(extent/2), int(extent/2), k, 0)

        # Output layer
        self.siu_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.siv_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.sic_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)        
        
        ##### Encoder 1 #####
        xe1_siu, xe1b_siu = self.siu_ec1(x) # SIU
        xe1_siv, xe1b_siv = self.siv_ec1(x) # SIV
        xe1_sic, xe1b_sic = self.sic_ec1(x) # SIC
        # Weighting block 1
        wb1_siu, wb1_siv, wb1_sic = self.wb1(xe1_siu, xe1_siv, xe1_sic)
        
        ##### Encoder 2 #####
        xe2_siu, xe2b_siu = self.siu_ec2(xe1_siu + wb1_siu) # SIU
        xe2_siv, xe2b_siv = self.siv_ec2(xe1_siv + wb1_siv) # SIV
        xe2_sic, xe2b_sic = self.sic_ec2(xe1_sic + wb1_sic) # SIC
        # Weighting block 2
        wb2_siu, wb2_siv, wb2_sic = self.wb2(xe2_siu, xe2_siv, xe2_sic)
        
        ##### Encoder 3 #####
        xe3_siu, xe3b_siu = self.siu_ec3(xe2_siu + wb2_siu) # SIU
        xe3_siv, xe3b_siv = self.siv_ec3(xe2_siv + wb2_siv) # SIV
        xe3_sic, xe3b_sic = self.sic_ec3(xe2_sic + wb2_sic) # SIC
        # Weighting block 3
        wb3_siu, wb3_siv, wb3_sic = self.wb3(xe3_siu, xe3_siv, xe3_sic)
        
        ##### Bottom bridge #####
        # SID
        xe3_siu = self.dropout(xe3_siu)
        xe41_siu = self.activation1(self.siu_ec41(xe3_siu + wb3_siu))
        xe42_siu = self.activation1(self.siu_ec42(xe41_siu))
        # SIV
        xe3_siv = self.dropout(xe3_siv)
        xe41_siv = self.activation1(self.siv_ec41(xe3_siv + wb3_siv))
        xe42_siv = self.activation1(self.siv_ec42(xe41_siv))
        # SIC
        xe3_sic = self.dropout(xe3_sic)
        xe41_sic = self.activation2(self.sic_ec41(xe3_sic + wb3_sic))
        xe42_sic = self.activation2(self.sic_ec42(xe41_sic))
        # output: 40x40x512
        # Weighting block 4
        wb4_siu, wb4_siv, wb4_sic = self.wb4(xe42_siu, xe42_siv, xe42_sic) 
        
        ##### Decoder 1 #####
        # SIU
        xd1_siu = self.siu_dc1(xe42_siu + wb4_siu, xe3b_siu)
        # SIV
        xd1_siv = self.siv_dc1(xe42_siv + wb4_siv, xe3b_siv)
        # SIC
        xd1_sic = self.sic_dc1(xe42_sic + wb4_sic, xe3b_sic)
        # Weighting block 5
        wb5_siu, wb5_siv, wb5_sic = self.wb5(xd1_siu, xd1_siv, xd1_sic) 
        
        ##### Decoder 2 #####
        # SIU
        xd2_siu = self.siu_dc2(xd1_siu + wb5_siu, xe2b_siu)
        # SIV
        xd2_siv = self.siv_dc2(xd1_siv + wb5_siv, xe2b_siv)
        # SIC
        xd2_sic = self.sic_dc2(xd1_sic + wb5_sic, xe2b_sic)
        # Weighting block 6
        wb6_siu, wb6_siv, wb6_sic = self.wb6(xd2_siu, xd2_siv, xd2_sic) 
        
        
        ##### Decoder 3 #####
        # SIU
        xd3_siu = self.siu_dc3(xd2_siu + wb6_siu, xe1b_siu)
        # SIV
        xd3_siv = self.siv_dc3(xd2_siv + wb6_siv, xe1b_siv)
        # SIC
        xd3_sic = self.sic_dc3(xd2_sic + wb6_sic, xe1b_sic)
        
        sic = torch.minimum(torch.maximum(self.sic_conv(xd3_sic), torch.tensor(0)), torch.tensor(1)) # ReLU
        siu = self.siu_conv(xd3_siu)
        siv = self.siv_conv(xd3_siv)
        
        # siu[sic == 0] = 0
        # siv[sic == 0] = 0
        
        out = torch.cat([siu, siv, sic], dim=1)
        out = out * (self.landmask == 0)

        return out
    
# Early branch UNET model
class EB_UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, k=3):
        super().__init__()
        
        self.activation = nn.Tanh()
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        ##### SIU BRANCH #####
        # input: 320x320x64
        self.siu_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siu_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siu_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siu_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siu_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siu_dc1 = decoder(512, 256) # output: 80x80x256
        self.siu_dc2 = decoder(256, 128) # output: 160x160x128
        self.siu_dc3 = decoder(128, 64) # output: 320x320x64     
        
        ##### SIV BRANCH #####
        # input: 320x320x64
        self.siv_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siv_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siv_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siv_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siv_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siv_dc1 = decoder(512, 256) # output: 80x80x256
        self.siv_dc2 = decoder(256, 128) # output: 160x160x128
        self.siv_dc3 = decoder(128, 64) # output: 320x320x64 
        
        ##### SIC BRANCH #####
        # input: 320x320x64
        self.sic_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.sic_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.sic_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.sic_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.sic_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.sic_dc1 = decoder(512, 256) # output: 80x80x256
        self.sic_dc2 = decoder(256, 128) # output: 160x160x128
        self.sic_dc3 = decoder(128, 64) # output: 320x320x64 

        # Output layer
        self.siu_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.siv_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.sic_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)        
        
        ##### Encoder 1 #####
        xe1_siu, xe1b_siu = self.siu_ec1(x) # SIU
        xe1_siv, xe1b_siv = self.siv_ec1(x) # SIV
        xe1_sic, xe1b_sic = self.sic_ec1(x) # SIC
        
        ##### Encoder 2 #####
        xe2_siu, xe2b_siu = self.siu_ec2(xe1_siu) # SIU
        xe2_siv, xe2b_siv = self.siv_ec2(xe1_siv) # SIV
        xe2_sic, xe2b_sic = self.sic_ec2(xe1_sic) # SIC

        ##### Encoder 3 #####
        xe3_siu, xe3b_siu = self.siu_ec3(xe2_siu) # SIU
        xe3_siv, xe3b_siv = self.siv_ec3(xe2_siv) # SIV
        xe3_sic, xe3b_sic = self.sic_ec3(xe2_sic) # SIC
        
        ##### Bottom bridge #####
        # SID
        xe41_siu = self.activation(self.siu_ec41(xe3_siu))
        xe42_siu = self.activation(self.siu_ec42(xe41_siu))
        # SIV
        xe41_siv = self.activation(self.siv_ec41(xe3_siv))
        xe42_siv = self.activation(self.siv_ec42(xe41_siv))
        # SIC
        xe41_sic = self.activation(self.sic_ec41(xe3_sic))
        xe42_sic = self.activation(self.sic_ec42(xe41_sic))
        # output: 40x40x512
        
        ##### Decoder 1 #####
        # SIU
        xd1_siu = self.siu_dc1(xe42_siu, xe3b_siu)
        # SIV
        xd1_siv = self.siv_dc1(xe42_siv, xe3b_siv)
        # SIC
        xd1_sic = self.sic_dc1(xe42_sic, xe3b_sic)
        # Weighting block 5
        
        ##### Decoder 2 #####
        # SIU
        xd2_siu = self.siu_dc2(xd1_siu, xe2b_siu)
        # SIV
        xd2_siv = self.siv_dc2(xd1_siv, xe2b_siv)
        # SIC
        xd2_sic = self.sic_dc2(xd1_sic, xe2b_sic)
        # Weighting block 6        
        
        ##### Decoder 3 #####
        # SIU
        xd3_siu = self.siu_dc3(xd2_siu, xe1b_siu)
        # SIV
        xd3_siv = self.siv_dc3(xd2_siv, xe1b_siv)
        # SIC
        xd3_sic = self.sic_dc3(xd2_sic, xe1b_sic)

        siu = self.siu_conv(xd3_siu)
        siv = self.siv_conv(xd3_siv)
        sic = self.sic_conv(xd3_sic)
        
        out = torch.cat([siu, siv, sic], dim=1)
        out = out * (self.landmask == 0)

        return out
    
# Early branch UNET model
class LB_UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, k=3):
        super().__init__()
        
        self.activation = nn.Tanh()
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        ##### SIU BRANCH #####
        # input: 320x320x64
        self.siu_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siu_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siu_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siu_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siu_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siu_dc1 = decoder(512, 256) # output: 80x80x256
        self.siu_dc2 = decoder(256, 128) # output: 160x160x128
        self.siu_dc3 = decoder(128, 64) # output: 320x320x64     

        # Output layer
        self.siu_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.siv_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        self.sic_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)        
        
        ##### Encoder 1 #####
        xe1_siu, xe1b_siu = self.siu_ec1(x) # SIU
        
        ##### Encoder 2 #####
        xe2_siu, xe2b_siu = self.siu_ec2(xe1_siu) # SIU

        ##### Encoder 3 #####
        xe3_siu, xe3b_siu = self.siu_ec3(xe2_siu) # SIU
        
        ##### Bottom bridge #####
        # SID
        xe41_siu = self.activation(self.siu_ec41(xe3_siu))
        xe42_siu = self.activation(self.siu_ec42(xe41_siu))
        # output: 40x40x512
        
        ##### Decoder 1 #####
        # SIU
        xd1_siu = self.siu_dc1(xe42_siu, xe3b_siu)
        # Weighting block 5
        
        ##### Decoder 2 #####
        # SIU
        xd2_siu = self.siu_dc2(xd1_siu, xe2b_siu)
        # Weighting block 6        
        
        ##### Decoder 3 #####
        # SIU
        xd3_siu = self.siu_dc3(xd2_siu, xe1b_siu)

        siu = self.siu_conv(xd3_siu)
        siv = self.siv_conv(xd3_siu)
        sic = self.sic_conv(xd3_siu)
        
        out = torch.cat([siu, siv, sic], dim=1)
        out = out * (self.landmask == 0)

        return out
    
# Information sharing UNET model
class IS_UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, extent, k=3):
        super().__init__()
        
        self.activation = nn.Tanh() #nn.LeakyReLU(0.1)
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        ##### SID BRANCH #####
        # input: 320x320x64
        self.sid_e11 = nn.Conv2d(32, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sid_e12 = nn.Conv2d(64, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sid_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x64

        # input: 160x160x64
        self.sid_e21 = nn.Conv2d(64, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sid_e22 = nn.Conv2d(128, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sid_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x80x128

        # input: 80x80x128
        self.sid_e31 = nn.Conv2d(128, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sid_e32 = nn.Conv2d(256, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sid_pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x40x256

        # input: 40x40x256
        self.sid_e41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.sid_e42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.sid_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # output: 80x80x256
        self.sid_d11 = nn.Conv2d(512, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sid_d12 = nn.Conv2d(256, 256, kernel_size=k, padding="same") # output: 80x80x256

        self.sid_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 160x160x128
        self.sid_d21 = nn.Conv2d(256, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sid_d22 = nn.Conv2d(128, 128, kernel_size=k, padding="same") # output: 160x160x128

        self.sid_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 320x320x64
        self.sid_d31 = nn.Conv2d(128, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sid_d32 = nn.Conv2d(64, 32, kernel_size=k, padding="same") # output: 320x320x64     
        
        
        ##### SIC BRANCH #####
        # input: 320x320x32
        self.sic_e11 = nn.Conv2d(32, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sic_e12 = nn.Conv2d(64, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sic_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x64

        # input: 160x160x64
        self.sic_e21 = nn.Conv2d(64, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sic_e22 = nn.Conv2d(128, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sic_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x80x128

        # input: 80x80x128
        self.sic_e31 = nn.Conv2d(128, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sic_e32 = nn.Conv2d(256, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sic_pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x40x256

        # input: 40x40x256
        self.sic_e41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.sic_e42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.sic_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # output: 80x80x256
        self.sic_d11 = nn.Conv2d(512, 256, kernel_size=k, padding="same") # output: 80x80x256
        self.sic_d12 = nn.Conv2d(256, 256, kernel_size=k, padding="same") # output: 80x80x256

        self.sic_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 160x160x128
        self.sic_d21 = nn.Conv2d(256, 128, kernel_size=k, padding="same") # output: 160x160x128
        self.sic_d22 = nn.Conv2d(128, 128, kernel_size=k, padding="same") # output: 160x160x128

        self.sic_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 320x320x64
        self.sic_d31 = nn.Conv2d(128, 64, kernel_size=k, padding="same") # output: 320x320x64
        self.sic_d32 = nn.Conv2d(64, 32, kernel_size=k, padding="same") # output: 320x320x64
        
        ##### Task Consistency Learning (TCL) Block #####
        self.tcl1 = TCL_block(64, int(extent/2), int(extent/2), k=3, w=0.0)        
        self.tcl2 = TCL_block(128, int(extent/4), int(extent/4), k=3, w=0.0)
        self.tcl3 = TCL_block(256, int(extent/8), int(extent/8), k=3, w=0.0)
        self.tcl4 = TCL_block(512, int(extent/8), int(extent/8), k=3, w=0.0)
        self.tcl5 = TCL_block(256, int(extent/4), int(extent/4), k=3, w=0.0)
        self.tcl6 = TCL_block(128, int(extent/2), int(extent/2), k=3, w=0.0)

        # Output layer
        self.sid_conv = nn.Conv2d(32, 2, kernel_size=k, padding="same")
        self.sic_conv = nn.Conv2d(32, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)
        
        ##### Encoder 1 #####
        # SID 
        xe11_sid = self.activation(self.sid_e11(x))
        xe12_sid = self.activation(self.sid_e12(xe11_sid))
        xp1_sid = self.sid_pool1(xe12_sid) # 160*160*64        
        # SIC
        xe11_sic = self.activation(self.sic_e11(x))
        xe12_sic = self.activation(self.sic_e12(xe11_sic))
        xp1_sic = self.sic_pool1(xe12_sic) # 160*160*64
        # TCL block 1
        tcl1_sid, tcl1_sic = self.tcl1(xp1_sid, xp1_sic)
        
        ##### Encoder 2 #####
        # SID 
        xe21_sid = self.activation(self.sid_e21(tcl1_sid + xp1_sid))
        xe22_sid = self.activation(self.sid_e22(xe21_sid))
        xp2_sid = self.sid_pool2(xe22_sid) # 80*80*128
        # SIC
        xe21_sic = self.activation(self.sic_e21(tcl1_sic + xp1_sic))
        xe22_sic = self.activation(self.sic_e22(xe21_sic))
        xp2_sic = self.sic_pool2(xe22_sic) # 80*80*128
        # TCL block 2
        tcl2_sid, tcl2_sic = self.tcl2(xp2_sid, xp2_sic)        
        
        ##### Encoder 3 #####
        # SID 
        xe31_sid = self.activation(self.sid_e31(tcl2_sid + xp2_sid))
        xe32_sid = self.activation(self.sid_e32(xe31_sid))
        xp3_sid = self.sid_pool3(xe32_sid) # 40*40*256
        # SIC
        xe31_sic = self.activation(self.sic_e31(tcl2_sic + xp2_sic))
        xe32_sic = self.activation(self.sic_e32(xe31_sic))
        xp3_sic = self.sic_pool3(xe32_sic) # 40*40*256
        # TCL block
        tcl3_sid, tcl3_sic = self.tcl3(xp3_sid, xp3_sic) 
        
        ##### Encoder 4 #####
        # SID
        xe41_sid = self.activation(self.sid_e41(tcl3_sid + xp3_sid))
        xe42_sid = self.activation(self.sid_e42(xe41_sid))
        # SIC
        xe41_sic = self.activation(self.sic_e41(tcl3_sic + xp3_sic))
        xe42_sic = self.activation(self.sic_e42(xe41_sic))
        # TCL block
        tcl4_sid, tcl4_sic = self.tcl4(xe42_sid, xe42_sic) 
        
        ##### Decoder 1 #####
        # SID
        xu1_sid = self.sid_upconv1(tcl4_sid + xe42_sid)
        xu11_sid = torch.cat([xu1_sid, xe32_sid], dim=1)
        xd11_sid = self.activation(self.sid_d11(xu11_sid))
        xd12_sid = self.activation(self.sid_d12(xd11_sid))
        # SIC
        xu1_sic = self.sic_upconv1(tcl4_sic + xe42_sic)
        xu11_sic = torch.cat([xu1_sic, xe32_sic], dim=1)
        xd11_sic = self.activation(self.sic_d11(xu11_sic))
        xd12_sic = self.activation(self.sic_d12(xd11_sic))
        # TCL block
        tcl5_sid, tcl5_sic = self.tcl5(xd12_sid, xd12_sic) 
        
        ##### Decoder 2 #####
        # SID
        xu2_sid = self.sid_upconv2(tcl5_sid + xd12_sid)
        xu22_sid = torch.cat([xu2_sid, xe22_sid], dim=1)
        xd21_sid = self.activation(self.sid_d21(xu22_sid))
        xd22_sid = self.activation(self.sid_d22(xd21_sid))
        # SIC
        xu2_sic = self.sic_upconv2(tcl5_sic + xd12_sic)
        xu22_sic = torch.cat([xu2_sic, xe22_sic], dim=1)
        xd21_sic = self.activation(self.sic_d21(xu22_sic))
        xd22_sic = self.activation(self.sic_d22(xd21_sic))
        # TCL block
        tcl6_sid, tcl6_sic = self.tcl6(xd22_sid, xd22_sic) 
        
        ##### Decoder 3 #####
        # SID
        xu3_sid = self.sid_upconv3(tcl6_sid + xd22_sid)
        xu33_sid = torch.cat([xu3_sid, xe12_sid], dim=1)
        xd31_sid = self.activation(self.sid_d31(xu33_sid))
        xd32_sid = self.activation(self.sid_d32(xd31_sid))
        # SIC
        xu3_sic = self.sic_upconv3(tcl6_sic + xd22_sic)
        xu33_sic = torch.cat([xu3_sic, xe12_sic], dim=1)
        xd31_sic = self.activation(self.sic_d31(xu33_sic))
        xd32_sic = self.activation(self.sic_d32(xd31_sic))

        sid = self.sid_conv(xd32_sid)
        sic = self.sic_conv(xd32_sic)
        
        out = torch.cat([sid, sic], dim=1)
        out = out * (self.landmask == 0)

        return out
    
# Hierarchical information sharing UNET model
class HIS_UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, landmask, extent, k=3):
        super().__init__()
        
        self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.landmask = landmask
        
        self.first_conv = nn.Conv2d(n_inputs, 32, kernel_size=k, padding="same")
        
        ##### SIU BRANCH #####
        # input: 320x320x64
        self.siu_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.siu_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.siu_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.siu_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.siu_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.siu_dc1 = decoder(512, 256) # output: 80x80x256
        self.siu_dc2 = decoder(256, 128) # output: 160x160x128
        self.siu_dc3 = decoder(128, 64) # output: 320x320x64     
        
        ##### SIC BRANCH #####
        # input: 320x320x64
        self.sic_ec1 = encoder(32, 64) # output: 160x160x64
        # input: 160x160x64
        self.sic_ec2 = encoder(64, 128) # output: 80x80x128
        # input: 80x80x128
        self.sic_ec3 = encoder(128, 256) # output: 40x40x256

        # input: 40x40x256
        self.sic_ec41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x512
        self.sic_ec42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x512

        # Decoder
        self.sic_dc1 = decoder(512, 256) # output: 80x80x256
        self.sic_dc2 = decoder(256, 128) # output: 160x160x128
        self.sic_dc3 = decoder(128, 64) # output: 320x320x64         
        
        ##### Weighting Blocks #####
        self.wb1 = AttBlock(64, int(extent/2), int(extent/2), k=3, w=0.1)        
        self.wb2 = AttBlock(128, int(extent/4), int(extent/4), k=3, w=0.1)
        self.wb3 = AttBlock(256, int(extent/8), int(extent/8), k=3, w=0.1)
        self.wb4 = AttBlock(512, int(extent/8), int(extent/8), k=3, w=0.1)
        self.wb5 = AttBlock(256, int(extent/4), int(extent/4), k=3, w=0.1)
        self.wb6 = AttBlock(128, int(extent/2), int(extent/2), k=3, w=0.1)

        # Output layer
        self.siu_conv = nn.Conv2d(64, 2, kernel_size=k, padding="same")
        self.sic_conv = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # First convolution
        x = self.first_conv(x)        
        
        ##### Encoder 1 #####
        xe1_siu, xe1b_siu = self.siu_ec1(x) # SIU
        xe1_sic, xe1b_sic = self.sic_ec1(x) # SIC
        # Weighting block 1
        wb1_siu, wb1_sic = self.wb1(xe1_siu, xe1_sic)
        
        ##### Encoder 2 #####
        xe2_siu, xe2b_siu = self.siu_ec2(wb1_siu) # SIU
        xe2_sic, xe2b_sic = self.sic_ec2(wb1_sic) # SIC
        # Weighting block 2
        wb2_siu, wb2_sic = self.wb2(xe2_siu, xe2_sic)
        
        ##### Encoder 3 #####
        xe3_siu, xe3b_siu = self.siu_ec3(wb2_siu) # SIU
        xe3_sic, xe3b_sic = self.sic_ec3(wb2_sic) # SIC
        # Weighting block 3
        wb3_siu, wb3_sic = self.wb3(xe3_siu, xe3_sic)
        
        ##### Bottom bridge #####
        # SID
        xe3_siu = self.dropout(xe3_siu)
        xe41_siu = self.activation1(self.siu_ec41(wb3_siu))
        xe42_siu = self.activation1(self.siu_ec42(xe41_siu))
        # SIC
        xe3_sic = self.dropout(xe3_sic)
        xe41_sic = self.activation1(self.sic_ec41(wb3_sic))
        xe42_sic = self.activation1(self.sic_ec42(xe41_sic))
        # output: 40x40x512
        # Weighting block 4
        wb4_siu, wb4_sic = self.wb4(xe42_siu, xe42_sic) 
        
        ##### Decoder 1 #####
        # SIU
        xd1_siu = self.siu_dc1(wb4_siu, xe3b_siu)
        # SIC
        xd1_sic = self.sic_dc1(wb4_sic, xe3b_sic)
        # Weighting block 5
        wb5_siu, wb5_sic = self.wb5(xd1_siu, xd1_sic) 
        
        ##### Decoder 2 #####
        # SIU
        xd2_siu = self.siu_dc2(wb5_siu, xe2b_siu)
        # SIC
        xd2_sic = self.sic_dc2(wb5_sic, xe2b_sic)
        # Weighting block 6
        wb6_siu, wb6_sic = self.wb6(xd2_siu, xd2_sic) 
        
        
        ##### Decoder 3 #####
        # SIU
        xd3_siu = self.siu_dc3(wb6_siu, xe1b_siu)
        # SIC
        xd3_sic = self.sic_dc3(wb6_sic, xe1b_sic)
        
        siu = self.siu_conv(xd3_siu)
        sic = self.sic_conv(xd3_sic)
                
        out = torch.cat([siu, sic], dim=1)
        out = out * (self.landmask == 0)

        return out
    

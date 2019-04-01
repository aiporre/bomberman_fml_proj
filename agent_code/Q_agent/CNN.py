import copy
import agent_code.Q_agent.config as c
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose
import numpy as np
from os.path import exists
from os import remove
from os.path import abspath, split



class NN(nn.Module):
    def __init__(self, num_out_channels=5, num_in_channels=1):
        super(NN, self).__init__()
        self.flat_feat_num = 1445 #84*84*num_in_channels  # 2592  # 256 #2*2*16
        self.num_in_channels = num_in_channels
        self.fc1 = nn.Linear(self.flat_feat_num, 120)
        self.fc2 = nn.Linear(120, 90)
        self.fc3 = nn.Linear(90, num_out_channels)

    def forward(self, x):
        x = x.float()
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class DuelingCNN(nn.Module):
    def __init__(self, state_height, state_width, num_out_channels=5, num_in_channels=1):
        super(DuelingCNN, self).__init__()
        self.state_height = state_height
        self.state_width = state_width
        self.num_in_channels = num_in_channels

        self.m = nn.Upsample(size=(84, 84), mode='nearest')

        self.conv1 = nn.Conv2d(num_in_channels, 16, 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=4)  # input 319x239 output 317x237
        self.conv3 = nn.Conv2d(32, 32, 5, stride=4)  # input 319x239 output 317x237
        self.flat_feat_num = 512 # 3136  # 2592  # 256 #2*2*16
        self.fc1_a = nn.Linear(self.flat_feat_num, 256)
        self.fc2_a = nn.Linear(256, num_out_channels)
        self.fc1_v = nn.Linear(self.flat_feat_num,256)
        self.fc2_v = nn.Linear(256, 1)


    def forward(self, x):
        x = x.float()
        # if self.num_in_channels == 1:
        #     x = x.view(-1, 1, self.state_height, self.state_width)
        x = self.m(x)
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(batch_size, -1)

        # Dueling layer
        a = F.relu(self.fc1_a(x))
        a = self.fc2_a(a)
        v = F.relu(self.fc1_v(x))
        v = self.fc2_v(v)
        # Merging layer
        x = v + a - torch.mean(a)
        return x


class CNN(nn.Module):
    def __init__(self, state_height, state_width, num_out_channels=5,num_in_channels=1):
        super(CNN, self).__init__()
        self.state_height = state_height
        self.state_width = state_width
        self.num_in_channels = num_in_channels

        # self.m = nn.Upsample(size=(84, 84), mode='nearest')
        self.conv1 = nn.Conv2d(num_in_channels, 16, 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=4)  # input 319x239 output 317x237
        self.conv3 = nn.Conv2d(32, 32, 5, stride= 4)  # input 319x239 output 317x237

        self.flat_feat_num = 512  # 288  # 2592  # 256 #2*2*16
        self.fc1 = nn.Linear(self.flat_feat_num, 256)
        self.fc2 = nn.Linear(256,num_out_channels)

        # self.conv1 = nn.Conv2d(num_in_channels, 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # input 319x239 output 317x237
        # self.conv3 = nn.Conv2d(64, 64, 3, stride= 1)  # input 319x239 output 317x237
        # self.flat_feat_num = 3136  # 288  # 2592  # 256 #2*2*16
        # self.fc1 = nn.Linear(self.flat_feat_num, 512)
        # self.prelu = nn.PReLU()
        # self.fc2 = nn.Linear(512,num_out_channels)

        # self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
        # self.conv2 = nn.Conv2d(16, 32, 4, stride=2)  # input 319x239 output 317x237
        # self.conv3 = nn.Conv2d(32, 32, 4, stride=2)  # input 319x239 output 317x237
        # self.fc1 = nn.Linear(self.flat_feat_num, num_out_channels)


    def forward(self, x):
        # if self.num_in_channels == 1:
        #     x = x.view(-1, 1, self.state_height, self.state_width)
        #     x = self.m(x)
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN17(nn.Module):
    def __init__(self, state_height, state_width, num_out_channels=5,num_in_channels=1):
        super(CNN17, self).__init__()
        self.state_height = state_height
        self.state_width = state_width
        self.num_in_channels = num_in_channels

        self.conv1 = nn.Conv2d(num_in_channels, 8, 2, stride=2)
        # self.pool1 = nn.MaxPool2d(2, 2)  # input 638x478 output 319x239
        self.conv2 = nn.Conv2d(8, 16, 2, stride=1)  # input 319x239 output 317x237
        # self.pool2 = nn.MaxPool2d(2, 2)  # input 317x237 output 158x118
        self.conv3 = nn.Conv2d(16, 16, 2, stride= 1)  # input 319x239 output 317x237
        self.flat_feat_num = 576  # 288  # 2592  # 256 #2*2*16
        self.fc1 = nn.Linear(self.flat_feat_num, num_out_channels)


    def forward(self, x):
        if self.num_in_channels == 1:
            x = x.view(-1, 1, self.state_height, self.state_width)
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)

        x = self.fc1(x)
        return x



class ModelBase(object):
    def __init__(self, transform, height, width, lr, num_out_channels, method='cnn', model_path=None, epochs = 1,
                 num_in_channels=1):
        self.transform = transform
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.num_out_channels = num_out_channels
        if method == 'cnn':
            network = CNN(height, width, num_out_channels=num_out_channels, num_in_channels=num_in_channels)
        elif method == 'dueling_cnn':
            network = DuelingCNN(height, width, num_out_channels=num_out_channels, num_in_channels=num_in_channels)
        elif method == 'nn':
            network = NN(num_out_channels=num_out_channels, num_in_channels=num_in_channels)
        elif method == 'cnn17':
            network = CNN17(height, width, num_out_channels=num_out_channels, num_in_channels=num_in_channels)
        else:
            raise ValueError('Method don\'t exist for CNN, change config.')
        if not exists(self.model_path):
            print('Creating new cnn...')
            print('target device is: ', self.device)
            # def init_weights(m):
            #     if type(m) == nn.Linear:
            #         torch.nn.init.xavier_uniform_(m.weight, gain=2.5)
            #         m.bias.data.fill_(0.5)
            # self.network.apply(init_weights)
            self.network = network.to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            print('Loading CNN from disk....')
            print('learning rate: ', self.lr)
            self.network = load_model(model_path, network)
            optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.optimizer = load_optimizer(model_path, optimizer)
            print('learning rate: ', self.lr)
        print('Is the network cuda?', next(self.network.parameters()).is_cuda)
        self.criterion = nn.MSELoss() # nn.SmoothL1Loss()  # nn.MSELoss()
        self.epochs = epochs

    def fit(self, states, actions, target_values):
        states = self.transform(states)
        print('fitting values: ')
        states = states.to(self.device)
        loss_val = 0
        for e in range(self.epochs):
            self.optimizer.zero_grad()
            predicted_values = self.network(states).gather(1, actions)
            loss = self.criterion(predicted_values, target_values.unsqueeze(1))
            loss.backward()
            for p in self.network.parameters():
                if p.grad is not None:
                    p.grad.data.clamp(-1, 1)
            self.optimizer.step()
            loss_val = loss.detach().cpu().numpy()
            print('LOSS = ', loss_val, 'EPOCH = ', e)
        print('====> last loss:', loss_val)

    def predict(self, states):
        with torch.no_grad():
            states = self.transform(states)
            states = states.to(self.device)
            return self.network(states)

    def save_model(self):
        if self.model_path:
            print('-------> save model', self.model_path)
            save_model(self.model_path, self.network, self.optimizer)
        else:
            print('-------> NOT SAVE MODEL model', self.model_path)


class ModelCoin(ModelBase):
    def __init__(self):
        model_path = abspath(c.coin['dql']['model_file'])
        transform = Compose([MultiToTensor(c.coin['network']['state_height'],
                                           c.coin['network']['state_width'],
                                           num_in_channels=c.coin['network']['num_in_channels'])])
        super(ModelCoin, self).__init__(transform, c.coin['network']['state_height'], c.coin['network']['state_width'],
                                        lr=c.coin['network']['lr'],
                                        num_out_channels=c.coin['network']['num_out_channels'],
                                        model_path=model_path,
                                        method=c.coin['network']['method'],
                                        num_in_channels=c.coin['network']['num_in_channels'])


class ModelExplorer(ModelBase):
    def __init__(self):
        model_path = abspath(c.explorer['dql']['model_file'])
        transform = Compose([MultiToTensor(c.explorer['network']['state_height'], c.explorer['network']['state_width'],
                                           num_in_channels=c.explorer['network']['num_in_channels']
                                           )])
        super(ModelExplorer, self).__init__(transform, c.explorer['network']['state_height'],
                                            c.explorer['network']['state_width'],
                                            lr=c.explorer['network']['lr'],
                                            num_out_channels=c.explorer['network']['num_out_channels'],
                                            model_path=model_path,
                                            method=c.explorer['network']['method'],
                                            num_in_channels = c.explorer['network']['num_in_channels'])


class ModelRunner(ModelBase):
    def __init__(self):
        model_path = abspath(c.runner['dql']['model_file'])
        transform = Compose([MultiToTensor(c.runner['network']['state_height'], c.runner['network']['state_width'],
                                           num_in_channels=c.runner['network']['num_in_channels']
                                           )])
        super(ModelRunner, self).__init__(transform, c.runner['network']['state_height'],
                                            c.runner['network']['state_width'],
                                            lr=c.runner['network']['lr'],
                                            num_out_channels=c.runner['network']['num_out_channels'],
                                            model_path=model_path,
                                            method=c.runner['network']['method'],
                                            num_in_channels = c.runner['network']['num_in_channels'])


class ModelMaster(ModelBase):
    def __init__(self):
        model_path = abspath(c.master['dql']['model_file'])
        transform = Compose([MultiToTensor(c.master['network']['state_height'], c.master['network']['state_width'],
                                           num_in_channels=c.master['network']['num_in_channels']
                                           )])
        super(ModelMaster, self).__init__(transform, c.master['network']['state_height'],
                                            c.master['network']['state_width'],
                                            lr=c.master['network']['lr'],
                                            num_out_channels=c.master['network']['num_out_channels'],
                                            model_path=model_path,
                                            method=c.master['network']['method'],
                                            num_in_channels=c.master['network']['num_in_channels'])


class FlatMatrixToTensor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        if isinstance(sample, list):
            output = np.empty((len(sample), self.height*self.width))
            for i in range(len(sample)):
                output[i] = np.fromstring(sample[i], dtype=int, sep=',')
        else:
            output = [np.fromstring(sample, dtype=int, sep=',')]
        return torch.tensor(output)


class DoubleToTensor(object):
    def __init__(self, height, width):
        assert width == height
        self.height = height
        self.width = width

    def __call__(self, sample):
        rept = np.ceil(84/self.width)
        subs = rept.astype(int)*self.width - 84
        if isinstance(sample, list):
            output = np.empty((len(sample), 2, 84, 84))
            for i in range(len(sample)):
                s = sample[i].split(';')
                u = np.fromstring(s[0], dtype=int, sep=',').reshape(self.height, self.width)
                v = np.fromstring(s[1], dtype=int, sep=',').reshape(self.height, self.width)
                output[i] = np.array([u.repeat(rept,axis=0).repeat(rept,axis=1)[:-subs,:-subs],
                                               v.repeat(rept,axis=0).repeat(rept,axis=1)[:-subs,:-subs]])
        else:
            sample = sample.split(';')
            u = np.fromstring(sample[0], dtype=int, sep=',').reshape(self.height, self.width)
            v = np.fromstring(sample[1], dtype=int, sep=',').reshape(self.height, self.width)
            output = np.array([u.repeat(rept,axis=0).repeat(rept,axis=1)[:-subs,:-subs],
                               v.repeat(rept,axis=0).repeat(rept,axis=1)[:-subs,:-subs]])
            output = np.expand_dims(output, axis=0)
        output = torch.tensor(output)
        return output.float()


class MultiToTensor(object):
    def __init__(self, height, width, num_in_channels=2):
        assert width == height
        self.height = height
        self.width = width
        self.channels = num_in_channels

    def __call__(self, sample):
        rept = np.ceil(84 / self.width)
        subs = rept.astype(int) * self.width - 84
        if isinstance(sample, list):
            output = np.empty((len(sample), self.channels, 84, 84))
            for i in range(len(sample)):
                s = sample[i].split(';')
                u = [
                    np.fromstring(s_i, dtype=int, sep=',').reshape(self.height, self.width).repeat(rept, axis=0).repeat(
                        rept, axis=1)[:-subs, :-subs] for s_i in s]
                output[i] = np.array(u)
        else:
            s = sample.split(';')
            u = [np.fromstring(s_i, dtype=int, sep=',').reshape(self.height, self.width).repeat(rept, axis=0).repeat(
                rept, axis=1)[:-subs, :-subs] for s_i in s]
            output = np.array(u)
            output = np.expand_dims(output, axis=0)
        output = torch.tensor(output)
        return output.float()

class MultiToTensorB(object):
    def __init__(self, height, width, num_in_channels=2):
        assert width == height
        self.height = height
        self.width = width
        self.channels = num_in_channels

    def __call__(self, sample):
        # rept = np.ceil(84 / self.width)
        # subs = rept.astype(int) * self.width - 84
        if isinstance(sample, list):
            output = np.empty((len(sample), self.channels, self.height, self.width))
            for i in range(len(sample)):
                s = sample[i].split(';')
                u = [
                    np.fromstring(s_i, dtype=int, sep=',').reshape(self.height, self.width) for s_i in s]
                output[i] = np.array(u)
        else:
            s = sample.split(';')
            u = [np.fromstring(s_i, dtype=int, sep=',').reshape(self.height, self.width) for s_i in s]
            output = np.array(u)
            output = np.expand_dims(output, axis=0)
        output = torch.tensor(output)
        return output.float()


class SampleToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if not isinstance(sample, list):
            output = [sample]
            return torch.tensor(output)
        return torch.tensor(sample)



def copy_model(model_qn, model_target):  # perhaps include in model
    # model_clone.load_state_dict()
    print('....COPYING NEW Q-DQN into the TARGET NETWORK')
    # Get parameter of Q-Network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parameters_qn = model_qn.network.state_dict()
    num_out_channels = model_qn.num_out_channels
    if not model_qn.num_out_channels == model_target.num_out_channels:
        raise ValueError(
            'Copy models error: models have not the same number of output channels: in model{}, out model {}'.format(
                model_qn.num_out_channels, model_target.num_out_channels))

    # concat_param = torch.cat(parameters_qm)
    num_in_channels = model_target.network.num_in_channels
    # cloning the q-network into the target network
    if model_target.network.__class__ == CNN:
        state_height = model_qn.network.state_height
        state_width = model_qn.network.state_width
        network = CNN(state_height, state_width, num_out_channels, num_in_channels=num_in_channels)
    elif model_target.network.__class__ == NN:
        network = NN(num_out_channels, num_in_channels=num_in_channels)
    elif model_target.network.__class__ == DuelingCNN:
        state_height = model_qn.network.state_height
        state_width = model_qn.network.state_width
        network = DuelingCNN(state_height, state_width, num_out_channels, num_in_channels=num_in_channels)
    elif model_target.network.__class__ == CNN17:
        state_height = model_qn.network.state_height
        state_width = model_qn.network.state_width
        network = CNN17(state_height, state_width, num_out_channels, num_in_channels=num_in_channels)
    else:
        raise ValueError('Class name doesn\'t exists, while copying model.')
    network.load_state_dict(parameters_qn)
    network.eval()
    model_target.network = network.to(device)


def save_model(path , model , optimizer=None):
    '''
    you can save a dict to be load as:
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    :param path:
    :return:
    '''
    print(" Saving model ... What's the path: ", path)
    source = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not exists(path):
        print('Error detected in the models directory... changing to abs-path mode: ')
        path = abspath(path)
        print(" Saving model ... What's the NEW path: ", path)

    if exists(path):
        remove(path)

    if optimizer:
        torch.save({
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': optimizer.state_dict() ,
        'source': source},
            path)
    else:
        torch.save({
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': None,
        'source': source},
            path)


def load_model(path, model):
    '''
    load a model in a form a dictionary and initialize the model and the optimizer
    :param source: it Is the device from where we are reading.
    :param target: it is the device in where we load the model.
    :param path: path where the model is stored
    :param model: model to be parametrized
    :param optimizer: optimizer to be parametrized/
    :return: model, optiizer,epoch and loss
    '''
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(target)
    checkpoint = torch.load(path, map_location=device)
    source = checkpoint['source']
    if source=='cpu' and target=='cpu':
        # CPU to CPU
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif not source == 'cpu' and target == 'cpu':
        # GPU to CPU
        checkpoint = torch.load(path , map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif not source == 'cpu' and not target == 'cpu':
        # GPU to GPU
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    else:
        # CPU to GPU
        checkpoint = torch.load(path , map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    return model

def load_optimizer(path, optimizer):
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(target)
    checkpoint = torch.load(path, map_location=device)

    if checkpoint['optimizer_state_dict']:
        source = checkpoint['source']
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(target)
        if source == 'cpu' and target == 'cpu':
            # CPU to CPU
            checkpoint = torch.load(path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif not source == 'cpu' and target == 'cpu':
            # GPU to CPU
            checkpoint = torch.load(path , map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif not source == 'cpu' and not target == 'cpu':
            # GPU to GPU
            checkpoint = torch.load(path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # CPU to GPU
            checkpoint = torch.load(path , map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer
    else:
        return optimizer

if __name__ == '__main__':
    model1 = ModelCoin()
    model2 = ModelCoin()
    copy_model(model1, model2)

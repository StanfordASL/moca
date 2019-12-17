import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
from metacpd.main.utils import listdir_nohidden
import torchvision

import os.path as osp
from PIL import Image

from torchvision import transforms

imagenet_train_superclasses = {
    0: [ 1, 2, 3, 4, 5, 6, 7, 8,19,20,21,48],
    1: [ 9,10,11,12,13,14,15,16,17,18],
    2: [22,26,27,38,39,46,50,53,54,57,62.63],
    3: [23,24,25,28,29,30,32,34,35,36,37,55,59,60,61,64],
    4: [31,33,40,41,42,43,44,45,47,49,51,52,56,58]
}

imagenet_val_superclasses = {
    0: [ 1, 4, 5],
    1: [ 2, 3],
    2: [ 6, 8,10,12,14,16],
    3: [13,15],
    4: [ 7, 9,11]
}

imagenet_test_superclasses = {
    0: [ 1, 2, 7, 8, 9],
    1: [ 3, 4, 5, 6],
    2: [13,16],
    3: [12,15,19,20],
    4: [10,11,14,17,18]
}

def uniform_sample(x):
    x_lower, x_upper = x
    return x_lower + np.random.rand()*(x_upper-x_lower)


class MiniImageNet(Dataset):
    def __init__(self, config, setname, path_prefix=None, data_length=1000000):
        
        ROOT_PATH = 'data/miniImageNet/'
        if path_prefix is not None:
            ROOT_PATH = path_prefix + ROOT_PATH
            

        self.horizon = config['data.horizon']
        self.switch_prob = config['data.hazard']
        self.data_length = data_length

        if setname == 'train':
            self.superclass_dict = imagenet_train_superclasses
        elif setname == 'val':
            self.superclass_dict = imagenet_val_superclasses
        elif setname == 'test':
            self.superclass_dict = imagenet_test_superclasses
        else:
            raise ValueError

        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        self.class_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        if setname == 'train' or setname == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif setname == 'test':
            self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.data_length

    def sample_subclasses(self):
        subclasses = []
        for i in range(5):
            subclasses.append(np.random.choice(self.superclass_dict[i])-1)

        return subclasses

    def get_sample_from_subclass(self,subclass_idx):
        imgs_per_class = 600
        return int(subclass_idx*imgs_per_class + np.random.randint(imgs_per_class))

    def sample(self, return_lists=True):
        x = np.zeros((self.horizon,3,84,84))
        y = np.zeros((self.horizon,5))

        subclasses_list = []
        switch_times = []
        for i in range(self.horizon):

            sampled_class = np.random.choice([0,1,2,3,4],p=self.class_probs)

            if np.random.rand() < self.switch_prob or i==0:
                subclasses = self.sample_subclasses()
                switch_times.append(i)

            subclasses_list.append(subclasses)

            img_idx = self.get_sample_from_subclass(subclasses[sampled_class])
            path = self.data[img_idx]

            x[i,:,:,:] = self.transform(Image.open(path).convert('RGB'))
            y[i,sampled_class] = 1

        if return_lists:
            return x,y,subclasses_list,switch_times

        return x,y

    def __getitem__(self,idx,train=True):
        # batching happens automatically

        data, labels, self.subclasses_list, self.switch_times = self.sample()

        switch_indicators = np.zeros(self.horizon)
        for i in range(self.horizon):
            if i in self.switch_times:
                switch_indicators[i] = 1.

        sample = {
            'x': data.astype(float),
            'y': labels.astype(float),
            'switch_times':switch_indicators.astype(float)
        }

        return sample

class SwitchingSinusoidDataset:
    def __init__(self, config, data_length=100000000):
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, 3.14]
        self.freq_range = [0.999, 1.0]
        self.x_range = [-5., 5.]
        self.horizon = config['data.horizon']
        self.switch_prob = config['data.hazard']
        
        self.data_length = data_length
        self.sigma_eps = eval(config['model.sigma_eps'])[0]
        
        self.noise_std = np.sqrt(self.sigma_eps)
        
    def __len__(self):
        return self.data_length-self.horizon
            
    def sample(self, return_lists=True):
        x_dim = 1 
        y_dim = 1
        
        x = np.zeros((self.horizon,x_dim))
        y = np.zeros((self.horizon,y_dim))

        amp_list = []
        phase_list = []
        freq_list = []
        switch_times = []
        for i in range(self.horizon):
            if np.random.rand() < self.switch_prob or i==0:
                amp = uniform_sample(self.amp_range)
                phase = uniform_sample(self.phase_range)
                freq = uniform_sample(self.freq_range)
                switch_times.append(i)
            
            amp_list.append(amp)
            phase_list.append(phase)
            freq_list.append(freq) 
            
            x_samp = uniform_sample(self.x_range)
            
            y_samp = amp*np.sin(freq*x_samp + phase) + self.noise_std*np.random.randn()

            x[i,0] = x_samp
            y[i,0] = y_samp

        if return_lists:
            return x,y,freq_list,amp_list,phase_list,switch_times

        return x,y
    
    def __getitem__(self,idx,train=True):
        # batching happens automatically
        
        data, labels, self.freq_list, self.amp_list, self.phase_list, self.switch_times = self.sample()
        switch_indicators = np.zeros(self.horizon)
        for i in range(self.horizon):
            if i in self.switch_times:
                switch_indicators[i] = 1.

        sample = {
            'x': data.astype(float),
            'y': labels.astype(float),
            'switch_times':switch_indicators.astype(float)
        }

        return sample

class RainbowMNISTDataset(Dataset):
    def __init__(self,config,path_prefix=None, train=True,data_length=1000000,path=None):

        self.data_length = data_length

        if path is None:
            if train == 'train':
                self.path = 'data/rainbow_mnist/train'
                print('Loading train set.')

            elif train == 'validate':
                self.path = 'data/rainbow_mnist/validate'
                print('Loading validation set.')

            elif train == 'test':
                self.path = 'data/rainbow_mnist/test'
                print('Loading test set.')

            else:
                raise ValueError
        else:
            raise NotImplementedError
        
        if path_prefix is not None:
            self.path = path_prefix + self.path

        self.horizon = config['data.horizon']
        self.switch_prob = config['data.hazard']

    def load_data(self,timeseries_length,path):
        labels = np.zeros((timeseries_length,10))
        data = np.zeros((timeseries_length,3,28,28)) 
        switch_times = []

        num_dirs = len(os.listdir(path))

        for i in range(timeseries_length):
            if np.random.rand() < self.switch_prob or i==0:
                # resample task (corresponding to resampling dir)

                task_dir = listdir_nohidden(path)
                task = np.random.choice(task_dir)
                task_path = path + '/' + task
                switch_times.append(i)

            # given task, sample digit
            full_path = task_path + '/' + np.random.choice(listdir_nohidden(task_path))

            digit = np.random.choice(listdir_nohidden(full_path))

            digit_path = full_path + '/' + digit

            # digit path is the label
            labels[i,int(digit)] = 1

            # load digit, add to time series
            final_path = np.random.choice(listdir_nohidden(digit_path))
            img = np.array(io.imread(digit_path + '/' + final_path)).transpose(2,0,1)
            data[i,:,:,:] = img
        return labels, data, switch_times

    def __len__(self):
        return self.data_length-self.horizon

    def __getitem__(self,idx,train=True):
        # batching happens automatically

        labels, data, self.switch_times = self.load_data(self.horizon,self.path)
        
        switch_indicators = np.zeros(self.horizon)
        for i in range(self.horizon):
            if i in self.switch_times:
                switch_indicators[i] = 1.

        sample = {
            'x': data.astype(float),
            'y': labels.astype(float),
            'switch_times':switch_indicators.astype(float)
        }
            
        return sample
    
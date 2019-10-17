
# coding: utf-8

# In[1]:


from scipy import signal as sg
from scipy import stats as st
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import h5py
import pandas as pd
import seaborn as sns
import torch
from scipy.io import loadmat


# In[2]:


class DataType():
    def __init__(self, data, tags, stimulus_type='None'):
        valid_stimulus = ['None', 'chirp', 'whitenoise', 'bar', 'color']
        
        if stimulus_type not in valid_stimulus:
            raise ValueError("{} not a valid stimulus ({})".format(stimulus_type, 
                                                                   ", ".join(valid_stimulus)))
        # Always [0] stimulus, [1] response, [2] extra, ...
        self.data = data
        self.tags = tags
        self.stimulus_dict = {}
        self.response_dict = {}
        self.avail_kwargs = {}
    
    def stimulus(self, cell_type=0, centered=False):
        pass
    
    def response(self, cell_type=0, centered=False):
        pass
    
    @staticmethod
    def memory(func1, func2, dictionary, to_check, if_boolean, *args):
        if to_check not in dictionary.keys():
            dictionary[to_check] = [None, None]
            if if_boolean:
                r = func1(*args)
                dictionary[to_check][0] = r
            else:
                r = func2(*args)
                dictionary[to_check][1] = r
        else:
            if if_boolean:
                if dictionary[to_check][0] is not None:
                    r = dictionary[to_check][0]
                else:
                    r = func1(*args)
                    dictionary[to_check][0] = r
            else:
                if dictionary[to_check][1] is not None:
                    r = dictionary[to_check][1]
                else:
                    r = func2(*args)
                    dictionary[to_check][1] = r
        return r, dictionary


# In[3]:


class Chirp(DataType):
    """
    Available kwargs for instance.stimulus('chirp') are:
        * torch = <bool> (default: torch=False)
        * batch = <Int> (default: batch=8)

    Available kwargs for instance.response('chirp') are:
        * torch = <bool> (default: torch=False)
        * batch = <Int> (default: batch=8
    """
    
    def __init__(self, data, tags):
        super(Chirp, self).__init__(data, tags, stimulus_type='chirp')
        
        #######
        self.avail_kwargs['stimulus'] = []
        self.avail_kwargs['response'] = ['cell_type']
        self.stim = data[0]
        self.resp = data[1]
        self.tags = tags
        #######
    
    def stimulus(self):
        return self.stim

    def response(self, cell_type=0, qi=False):
        # added memory, not sure if we should keep it, maybe toggle it when creating the class
        resp = self.resp[self.tags[cell_type]]
        return resp


# In[7]:


class WhiteNoise(DataType):
    def __init__(self, data, tags, qi, cell='bipolar'):
        super(WhiteNoise, self).__init__(data, tags, stimulus_type='whitenoise')
        self.avail_kwargs['stimulus'] = ['cell_type', 'centered', 'loc', 'qi']
        self.avail_kwargs['response'] = ['cell_type', 'centered', 'qi']
        self.cell = cell
        self.tags = tags
        self.qi = qi
        
        if self.cell != 'bipolar':
            # Fix MaGiC NUmbErs
            self.stim = data[0]
            self.resp = data[1]
            self.stim = np.repeat(self.stim[:,:,31:1632,3], 64, axis=2)[:,:,::5]
            self.stim = np.swapaxes(self.stim, 0, -1)
        else:
            self.stim = data[0]
            self.stim = np.repeat(self.stim[:1450,:,:], 64, axis=0)[::5,:,:]
            self.resp = data[1]
            self.stim_time = data[2]
            
    def calculate_qi(self, qi, cell_type=0):
        cells = self.tags[cell_type]
        w = np.where(self.qi[cells] >= qi, 
                     self.qi[cells], 0).reshape(-1)
        w = np.argwhere(w).reshape(-1)
        return w

    def response(self, cell_type=0, centered=False, qi=0):
        cells = self.tags[cell_type]
        if self.cell == 'bipolar':
            resp = np.zeros((len(cells), 18560))
            for idx, cell in enumerate(cells):
                start = int(self.stim_time[cell][0] * 31.25)
                wn_rs[idx] = sg.resample_poly(self.resp[cell][start:], 256, 125)[:18560]
        else:
            resp = np.zeros((len(cells), 20493))
            for idx, cell in enumerate(cells):
                resp[idx] = sg.resample_poly(self.resp[31:1632, cell], 64, 5)
        if qi > 0:
            w = self.calculate_qi(qi, cell_type=cell_type)
            resp = resp[w]
        return resp
    
    def stimulus(self, cell_type=0, centered=False, loc=None, qi=0):
        if not centered:
            stim = self.stim
        else:
            if cell_type==0:
                raise MemoryError("Transforming all the cell types takes too much memory, please use a cell type (cell_type=int)")
            if loc is None:
                raise ValueError("You need to enter the a `loc` dictionary, to obtain it please do instance.rf(centered=True)")
            
            if qi > 0:
                w = self.calculate_qi(qi, cell_type=cell_type)
                cells = self.tags[cell_type][w]
            else:
                cells = self.tags[cell_type]
            
            data_to_center = np.array(self.stim)
            #centered_cells = np.zeros((*cells.shape, *data_to_center.shape))
            centered_shape = np.zeros((*cells.shape, data_to_center.shape[0], 7, 7))
            
            cell_type = loc['cell_type']

            for cell in range(len(cells)):
                to_center = data_to_center
                shifted = np.roll(to_center, (loc[cell][0],loc[cell][1]), axis=(1,2))
                centered_shape[cell, :, :, :] = shifted[:,7:14,5:12]
            stim = centered_shape
                        
        return stim


# In[5]:


class Rf(DataType):
    def __init__(self, data, tags, qi, cell="bipolar"):
        super(Rf, self).__init__(data, tags)
        self.rf_dict = {}
        self.avail_kwargs['rf'] = ['cell_type', 'centered', 'cell', 'qi']
        self.data = data
        self.cell=cell
        self.qi = qi
    
    def calculate_qi(self, qi, cell_type=0):
        cells = self.tags[cell_type]
        w = np.where(self.qi[cells] >= qi, 
                     self.qi[cells], 0).reshape(-1)
        w = np.argwhere(w).reshape(-1)
        return w
    
    def return_rf(self, cell_type=0, centered=False, qi=0):
        if centered:
            data_to_center = np.array(self.data)[self.tags[cell_type]]
            centered_cells = np.zeros(data_to_center.shape)
            centered_shape = np.zeros((data_to_center.shape[0], 7, 7))
            center = (8, 10)
            loc = {'center':center, 'cell_type':cell_type}
            for i in range(data_to_center.shape[0]):
                to_center = data_to_center[i]
                max_location = np.unravel_index(to_center.argmax(), to_center.shape)
                x_distance = center[0] - max_location[0]
                y_distance = center[1] - max_location[1]
                shifted_x = np.roll(to_center, x_distance, axis=0)
                shifted_y = np.roll(shifted_x, y_distance, axis=1)
                centered_cells[i, :, :] = shifted_y
                centered_shape[i, :, :] = centered_cells[i, 5:12,7:14]
                loc[i] = (x_distance, y_distance)
            rf = centered_shape
        else:
            rf = np.array(self.data)[self.tags[cell_type]]
            loc = None
            
        if qi > 0:
            w = self.calculate_qi(qi, cell_type=cell_type)
            rf = rf[w]
        
        return rf, loc


# In[6]:


class Data():   
    def __init__(self, directory, cell="bipolar"):
        # Error handling
        if cell not in ["bipolar", "ganglionar"]:
            raise ValueError("cell value should be \"bipolar\" or \"ganglionar\" you tried \"{}\".".format(cell))
        
        self.errordict = {'bipolartype':"Types admitted 0..15, (0 for all types, 15 for NaN) you tried cell type {}",
                    'ganglionartype':"Types admitted 0..40, (0 for all types, 40 for NaN) you tried cell type {}",}
            
        # Continue the initialization
        self.directory = directory
        self.cell = cell
        
        if self.cell == "bipolar":
            self.file_name = 'FrankeEtAl_BCs_2017_v1.mat'
            self.data = h5py.File(self.directory + self.file_name, 'r')
            self.white_noise_data = h5py.File(self.directory + 'FrankeEtAl_BCs_2017_noise_raw.mat', 'r')
            
        else:
            self.file_name = 'BadenEtAl_RGCs_2016_v1.mat'
            self.data = loadmat(self.directory + self.file_name)
            self.white_noise_data = self.data
        
        # Shared parameters
        
        self.params = {'chirp': [None, self._generate_chirp],
                           'whitenoise': [None, self._generate_whitenoise],
                           'bar': [None, self._generate_bar],
                           'color': [None, self._generate_color],
                           'rf': [None, self._generate_rf],}
        
        self.tags = {}
      
        if cell == "bipolar":
            max_ = int(np.where(np.isnan(self.data['cluster_idx'][0]), -1, self.data['cluster_idx'][0]).max())
            for i in range(1, max_ + 1):
                self.tags[i] = np.argwhere(self.data['cluster_idx'][0] == i)[:,0]
                tags = np.array(self.data['cluster_idx'][0])
            self.tags[max_+1] = np.argwhere(np.isnan(self.data['cluster_idx'][0]))[:,0]
            self.tags[0] = np.arange((self.data['cluster_idx'].shape[-1]))
        else:
            max_ = int(np.where(np.isnan(self.data['cluster_idx']), -1, self.data['cluster_idx']).max())
            for i in range(1, max_ + 1):
                self.tags[i] = np.argwhere(self.data['cluster_idx'] == i)[:,0]
                tags = np.array(self.data['cluster_idx'])
            self.tags[max_+1] = np.argwhere(np.isnan(self.data['cluster_idx']))[:,0]
            self.tags[0] = self.data['cluster_idx'].reshape(-1)
        
    def __init(self, stimulus, stimulus_type, kwargs_dict):
        if stimulus not in self.params:
            raise ValueError("{} not a valid stimulus ({})"
                             .format(stimulus, ', '.join(self.params)))
        
        if self.params[stimulus][0] == None:
            data = self.data
            if stimulus == "whitenoise":
                data_wn = self.white_noise_data
                init = self.params[stimulus][1](self.cell, data_wn, data, self.tags)
            else:
                init = self.params[stimulus][1](self.cell, data, self.tags)
            self.params[stimulus][0] = init
        
        if "cell_type" in kwargs_dict:
            if self.cell == 'bipolar':
                if kwargs_dict["cell_type"] not in range(16):
                    raise ValueError(self.errordict['bipolartype'].format(kwargs_dict["cell_type"]))
            else:
                if kwargs_dict["cell_type"] not in range(41):
                    raise ValueError(self.errordict['ganglionartype'].format(kwargs_dict["cell_type"]))
        
        l1 = self.params[stimulus][0].avail_kwargs[stimulus_type]
        d1 = kwargs_dict
        
        if len(d1) > len(l1):
            raise TypeError("Too many keyword arguments (Expected up to {} got {})".format(len(l1), len(d1)))

        if False in [list(d1.keys())[i] in l1 for i in range(len(d1))]:
            raise TypeError("Available keyword arguments for {} with {} are {}".format(stimulus_type, stimulus, ", ".join(l1)))        
        
    def response(self, stimulus, **kwargs):
        # Check errors
        self.__init(stimulus, "response", kwargs)
                
        r = self.params[stimulus][0].response(**kwargs)
        
        return r
        
    def stimulus(self, stimulus, **kwargs):
        # Check errors
        self.__init(stimulus, "stimulus", kwargs)

        s = self.params[stimulus][0].stimulus(**kwargs)
    
        return s
    
    def rf(self, **kwargs):
        # Check errors
        self.__init('rf', "rf", kwargs)
    
        r, loc = self.params['rf'][0].return_rf(**kwargs)
        
        if "centered" in kwargs:
            return r, loc
        else:
            return r
        
            
    @staticmethod
    def _generate_chirp(cell, data, tags):
        if cell == "bipolar":
            chirp_stim = data['chirp_stim'][0]
            chirp_resp = np.array(data['gchirp_avg'])
            chirp_resp_time = data['chirp_time']

            chirp_stim_resampled = sg.resample_poly(chirp_stim, 995, 15994, window=('kaiser', 500.0))
            chirp_stim_resampled = np.concatenate((chirp_stim_resampled, np.zeros(2048 - 1990)))
            chirp_response_resampled = chirp_resp

        else:
            chirp_stim = data['chirp_stim']
            chirp_resp = data['chirp_avg']
            chirp_resp_time = data['chirp_time']

            # Resampled
            chirp_stim_resampled = sg.resample_poly(chirp_stim, 2048, 31988, window=('kaiser', 680.0)).reshape(-1)          
            chirp_response_resampled = np.zeros((chirp_resp.shape[1], 2048))

            for x in range(chirp_resp.shape[1]):
                chirp_response_resampled[x] = sg.resample(chirp_resp[:,x], 2048, t = chirp_resp_time[0], axis = 0)[0]

        return Chirp((chirp_stim_resampled, chirp_response_resampled), tags)
    
    @staticmethod
    def _generate_rf(cell, data, tags):
        rf_data = data['rf_map']
        rf_qi = np.array(data['rf_qi']).reshape(-1)
        if cell != 'bipolar':
            rf_data = np.swapaxes(data['rf_map'], 0, 2)
        return Rf(rf_data, tags, rf_qi, cell=cell)
    
    @staticmethod
    def _generate_whitenoise(cell, data_wn, data, tags):
        stim = 'noise_movie' if cell == 'bipolar' else 'noise_stim'
        stim_time = 'noise_stim_time' if cell == 'bipolar' else None
        resp = 'noise_time' if cell == 'bipolar' else 'noise_trace'
        wn_data = np.array(data_wn[stim])
        wn_response = np.array(data_wn[resp])
        wn_stim_time = np.array(data_wn.get(stim_time, 0))
        wn_qi = np.array(data['rf_qi']).reshape(-1)
        
        return WhiteNoise((wn_data, wn_response, wn_stim_time), tags, wn_qi, cell=cell)
        
    @staticmethod
    def _generate_bar(cell, data, tags):
        raise NotImplementedError("Bar noise not yet implemented.")
        
    @staticmethod
    def _generate_color(cell, data, tags):
        raise NotImplementedError("Color not yet implemented.")
    
    def keys(self):
        both = ['chirp()', 'whitenoise()', 'bar()', 'color()', 'tags']
        if self.cell == "bipolar":
            both += []
        else:
            both += []
        return both


import numpy as np
import librosa.output
import librosa
from intervaltree import Interval,IntervalTree
from scipy import fft 
import pickle
import h5py
import sys


class hyperparams(object):
    def __init__(self):
        self.sr = 44100 # Sampling rate.
        self.n_fft = 2048 # fft points (samples)
        self.stride = 256 # 256 samples hop between windows    
        self.wps = 44100 // 256 # ~86 windows/second
        self.instrument = { 
                            'cello': [2217, 2218, 2219, 2220 ,2221, 2222, 2293, 2294, 2295, 2296, 2297, 2298],
                            'violin': [2191, 2244, 2288, 2289, 2659], 
                            'flute':[2202, 2203, 2204]
                            }
        self.hop_inst = {'cello': self.wps, 'violin': int(self.wps * 0.5), 'flute': int(self.wps*0.25)}
                    

hp = hyperparams()


def get_data(): 
    '''
    
    Extract the desired solo data from the dataset.
    
    Default: 
        Process cello, violin, flute 
    
    '''
    dataset = np.load(open('data/musicnet.npz','rb'), encoding = 'latin1')
    train_data = h5py.File('data/train_data.hdf5', 'w') 

    for inst in hp.instrument:
        print ('------ Processing ' + inst + ' ------')
        score = []
        audio = []
        for song in hp.instrument[inst]: 
            a,b = dataset[str(song)] 
            score.append(a)
            audio.append(b)

        spec_list, score_list, onoff_list = process_data(score,audio,inst)   
        train_data.create_dataset(inst + "_spec", data=spec_list)
        train_data.create_dataset(inst + "_pianoroll", data=score_list)
        train_data.create_dataset(inst + "_onoff", data=onoff_list)  


def process_data(X, Y, inst):
    '''
    Data Pre-processing
        
    Score: 
        Generate pianoroll from interval tree data structure
    
    Audio: 
        Convert waveform into power spectrogram

    '''
    def process_spectrum(X, step, hop):
        audio = X[i][(step * hop * hp.stride): (step * hop * hp.stride) + ((hp.wps*5 - 1)* hp.stride)] 
        spec = librosa.stft(audio, n_fft= hp.n_fft, hop_length = hp.stride)
        magnitude = np.log1p(np.abs(spec)**2)
        return magnitude

    def process_score(Y, step, hop):
        score = np.zeros((hp.wps*5, 128))  
        onset = np.zeros(score.shape)    
        offset = np.zeros(score.shape) 

        for window in range(score.shape[0]):
            
            #For score, set all notes to 1 if they are played at this window timestep 
            labels = Y[i][(step * hop + window) * hp.stride] 
            for label in labels: 
                score[window,label.data[1]] = 1 
        
        
            #For onset/offset, set onset to 1 and offset to -1 
            if window != 0:
                onset[window][np.setdiff1d(score[window].nonzero(), score[window-1].nonzero())] = 1
                offset[window][np.setdiff1d(score[window-1].nonzero(), score[window].nonzero())] = -1                    
            else:
                onset[window][score[window].nonzero()] = 1
        
        
        onset += offset 
        return score, onset
    

    spec_list=[]
    score_list=[]
    onoff_list=[]
    num_songs = len(X)
    hop = hp.hop_inst[inst]
    for i in range(num_songs):
        song_length = len(X[i])
        num_spec = (song_length) // (hop * hp.stride) 
        print ('{} song {} has {} windows'.format(inst, i, num_spec))

        for step in range(num_spec - 30):
            if step % 50 == 0:
                print ('{} steps of {} song {} has been done'.format(step,inst,i))        
            spec_list.append(process_spectrum(X,step,hop))
            score, onoff = process_score(Y,step,hop)
            score_list.append(score)
            onoff_list.append(onoff)

    return np.array(spec_list), np.array(score_list), np.array(onoff_list)

    




def main():  
    get_data()
   

if __name__ == "__main__":
    main()
    

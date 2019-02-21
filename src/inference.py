import torch
import pretty_midi
import numpy as np
import h5py
import pickle
import torch.nn as nn
import torch.utils.data as utils
import json
import os
from model import PerformanceNet
import librosa
from tqdm import tqdm
import sys
class AudioSynthesizer():
    def __init__(self, checkpoint, exp_dir, data_source):
        self.exp_dir = exp_dir
        self.checkpoint = torch.load(os.path.join(exp_dir,checkpoint))
        self.sample_rate = 44100
        self.wps = 44100//256
        self.data_source = data_source
                
    def get_test_midi(self):

        X = np.load(os.path.join(self.exp_dir,'test_data/test_X.npy'))
        rand = np.random.randint(len(X),size=5)
        score = [X[i] for i in rand]
        return torch.Tensor(score).cuda()

    def process_custom_midi(self, midi_filename):

        midi_dir = os.path.join(self.exp_dir,'midi')
        midi = pretty_midi.PrettyMIDI(os.path.join(midi_dir,midi_filename))    
        pianoroll = midi.get_piano_roll(fs=self.wps).T
        pianoroll[pianoroll.nonzero()] = 1
        onoff = np.zeros(pianoroll.shape) 
        for i in range(pianoroll.shape[0]):
            if i == 0:
                onoff[i][pianoroll[i].nonzero()] = 1
            else:
                onoff[i][np.setdiff1d(pianoroll[i-1].nonzero(), pianoroll[i].nonzero())] = -1
                onoff[i][np.setdiff1d(pianoroll[i].nonzero(), pianoroll[i-1].nonzero())] = 1 
        
        return pianoroll, onoff


    def inference(self):
        model = PerformanceNet().cuda()
        model.load_state_dict(self.checkpoint['state_dict'])

        if self.data_source == 'TEST_DATA':
            score = self.get_test_midi()
            score, onoff = torch.split(score, 128, dim=1)
        else:
            score, onoff = self.process_custom_midi(self.data_source)
                   
        print ('Inferencing spectrogram......')

        with torch.no_grad():
            model.eval()    
            test_results = model(score, onoff)
            test_results = test_results.cpu().numpy()
 
        output_dir = self.create_output_dir()

        for i in range(len(test_results)):
            audio = self.griffinlim(test_results[i], audio_id = i+1)
            librosa.output.write_wav(os.path.join(output_dir,'output-{}.wav'.format(i+1)), audio, self.sample_rate)
    
    def create_output_dir(self):
        success = False
        dir_id = 1
        while not success:
            try:
                audio_out_dir = os.path.join(self.exp_dir,'audio_output_{}'.format(dir_id))
                os.makedirs(audio_out_dir)
                success = True
            except FileExistsError:
                dir_id += 1
        return audio_out_dir

    def griffinlim(self, spectrogram, audio_id, n_iter = 300, window = 'hann', n_fft = 2048, hop_length = 256, verbose = False):
        
        print ('Synthesizing audio {}'.format(audio_id))

        if hop_length == -1:
            hop_length = n_fft // 4
            spectrogram[0:5] = 0

        spectrogram[150:] = 0
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

        t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
        for i in t:
            full = np.abs(spectrogram).astype(np.complex) * angles
            inverse = librosa.istft(full, hop_length = hop_length, window = window)
            rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
            angles = np.exp(1j * np.angle(rebuilt))

            if verbose:
                diff = np.abs(spectrogram) - np.abs(rebuilt)
                t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)

        return inverse


def main():
    exp_dir = os.path.join(os.path.abspath('./experiments'), sys.argv[1]) # which experiment to test
    data_source = sys.argv[2] # test with testing data or customized data    
    with open(os.path.join(exp_dir,'hyperparams.json'), 'r') as hpfile:
        hp = json.load(hpfile)
    checkpoints = 'checkpoint-{}.tar'.format(hp['best_epoch'])
    AudioSynth = AudioSynthesizer(checkpoints, exp_dir, data_source) 
    AudioSynth.inference()


if __name__ == "__main__":
    main()
            


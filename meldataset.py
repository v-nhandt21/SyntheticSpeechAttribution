import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import torchaudio
from utils import AttrDict, build_env
MAX_WAV_VALUE = 32768.0
import matplotlib
matplotlib.use('TKAgg')

def load_wav(full_path):
     sampling_rate, data = read(full_path)
     return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
     return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
     return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
     output = dynamic_range_compression_torch(magnitudes)
     return output


def spectral_de_normalize_torch(magnitudes):
     output = dynamic_range_decompression_torch(magnitudes)
     return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, h, center=False):
     if torch.min(y) < -1.:
          print('min value is ', torch.min(y))
     if torch.max(y) > 1.:
          print('max value is ', torch.max(y))
     global mel_basis, hann_window
     if h.fmax not in mel_basis:
          mel = librosa_mel_fn(h.sampling_rate, h.n_fft, h.num_mels, h.fmin, h.fmax)
          mel_basis[str(h.fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
          hann_window[str(y.device)] = torch.hann_window(h.win_size).to(y.device)
     y = torch.nn.functional.pad(y.unsqueeze(1), (int((h.n_fft-h.hop_size)/2), int((h.n_fft-h.hop_size)/2)), mode='reflect')
     y = y.squeeze(1)
     spec = torch.stft(y, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=hann_window[str(y.device)],
                         center=center, pad_mode='reflect', normalized=False, onesided=True)
     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
     spec = torch.matmul(mel_basis[str(h.fmax)+'_'+str(y.device)], spec)
     spec = spectral_normalize_torch(spec)

     return spec


def get_dataset_filelist(a, class5=False):
     with open(a.input_training_file, 'r', encoding='utf-8') as fi:
          if class5:
               training_files = [x.split(',') for x in fi.read().split('\n') if int(x.split(',')[1]) != 5]
          else:
               training_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]
     with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
          if class5:
               validation_files = [x.split(',') for x in fi.read().split('\n') if int(x.split(',')[1]) != 5]
          else:
               validation_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]

     return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
     def __init__(self, training_files, h):
          self.audio_files = training_files
          random.seed(1234)
          random.shuffle(self.audio_files)
          self.segment_size = h.segment_size
          self.h = h
          self.specaugment = h.specaugment
          self.masking_time = torchaudio.transforms.TimeMasking(time_mask_param=h.time_mask_param) #5
          self.masking_freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=h.freq_mask_param) #2

     def __getitem__(self, index):

          filename, y = self.audio_files[index]
          mel = self.get_feature(filename)

          if self.specaugment == 1:
               mel = self.masking_time(mel)
               mel = self.masking_freq(mel)

          return (mel.squeeze(), int(y))

     def extract_mel(self, save_path):
          for file, _ in self.audio_files:
               mel = self.get_feature(file)
               np.save(save_path + "/"+file.split("/")[-1] ,mel.detach().numpy())

     def get_feature(self, filename):
          
          audio, sampling_rate = load_wav(filename)
          audio = audio / MAX_WAV_VALUE
          audio = normalize(audio) * 0.95

          audio = torch.FloatTensor(audio)
          audio = audio.unsqueeze(0)

          if audio.size(1) >= self.segment_size:
               max_audio_start = audio.size(1) - self.segment_size
               audio_start = random.randint(0, max_audio_start)
               audio = audio[:, audio_start:audio_start+self.segment_size]
          else:
               audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

          mel = mel_spectrogram(audio, self.h)
          
          return mel

     def __len__(self):
          return len(self.audio_files)

if __name__ == '__main__':
     import argparse, json, shutil
     device = "cuda"

     parser = argparse.ArgumentParser()
     parser.add_argument('--config', default='config.json')
     parser.add_argument('--input_testing_file', default='/home/nhandt23/Desktop/SPCUP/Data/val.txt')

     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     json_config = json.loads(data)
     h = AttrDict(json_config)

     with open(a.input_testing_file, 'r', encoding='utf-8') as fi:
          testing_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]

     dataset = MelDataset(testing_files, h)
     dataset.extract_mel("Mel")
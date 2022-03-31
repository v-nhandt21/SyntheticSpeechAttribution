import glob
import os
import argparse
import json
import torch
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import shutil
import onnxruntime
import numpy as np
import random
import sys

class AttrDict(dict):
         def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self


def build_env(config, config_name, path):
     t_path = os.path.join(path, config_name)
     if config != t_path:
          os.makedirs(path, exist_ok=True)
          shutil.copyfile(config, os.path.join(path, config_name))

h = None
device = None
from efficientnet_pytorch import EfficientNet


def export_onnx(model_path, outpath_onnx, static=True):
     generator = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to("cuda")

     state_dict_g = torch.load(model_path, map_location="cuda")

     generator.load_state_dict(state_dict_g['generator'])

     generator.eval()

     # Dummy input for ONNX
     x = torch.randn(1, 3, 80, 93, requires_grad=True).to("cuda")

     generator.set_swish(memory_efficient=False) # ?

     if static:
          torch.onnx.export(generator, x, outpath_onnx, verbose=False)
     else:
          torch.onnx.export(generator,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    outpath_onnx,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {3 : 'seq_length'},    # variable lenght axes
                                   'output' : {1 : 'seq_length'}} , verbose=False)

def inference_onnx(path="Static/sccup.onnx", static=True):

     config_file = 'config.json'
     with open(config_file) as f:
          data = f.read()

     global h
     json_config = json.loads(data)
     h = AttrDict(json_config)

     torch.manual_seed(h.seed)
     global device
     if torch.cuda.is_available():
          torch.cuda.manual_seed(h.seed)
          device = torch.device('cuda')
     else:
          device = torch.device('cpu')

     model = onnxruntime.InferenceSession(path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
     Count = 0
     with torch.no_grad():
          with open("Static/data/val.txt", "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in lines:
                    file, label = line.split(",")

                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    audio = torch.FloatTensor(wav).to(device).unsqueeze(0)

                    if static:
                         max_audio_start = audio.size(1) - 24000
                         audio_start = random.randint(0, max_audio_start)
                         audio = audio[:, audio_start:audio_start+24000]

                    x = mel_spectrogram(audio, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                    x = x.unsqueeze(1).repeat(1,3,1,1).detach().cpu().numpy()

                    output = model.run(None, {model.get_inputs()[0].name: x})

                    predict = np.argmax(output[0][0])

                    if int(predict) == int(label):
                         Count += 1
                    else:
                         print(line)

               print(Count/len(lines))

if __name__ == '__main__':

     if len(sys.argv) > 1:
          model_path = sys.argv[1]
     else:
          model_path = "Static/checkpoint/g_00015000"

     outpath_onnx = "Static/sccup.onnx"
     
     export_onnx(model_path, outpath_onnx)

     # Try to test model in onnx
     inference_onnx(outpath_onnx)
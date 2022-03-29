import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
from meldataset import MelDataset, mel_spectrogram, MAX_WAV_VALUE, load_wav
from utils import save_checkpoint
import os
from utils import AttrDict, save_checkpoint
from tqdm import tqdm
from model import Classifier, Cascade
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def get_mel(x, h):
     return mel_spectrogram(x, h)

def predict_cascade(part, model_path_unseen, model_path_algo, h):
     model_unseen = Cascade(unseen=True, h = h).to("cuda")
     model_algo = Cascade(unseen=False, h = h).to("cuda")

     model_unseen.load_state_dict(torch.load(model_path_unseen, map_location="cuda")['generator'])
     model_unseen.eval()

     model_algo.load_state_dict(torch.load(model_path_algo, map_location="cuda")['generator'])
     model_algo.eval()

     data = MelDataset([], h)
     softmax = torch.nn.Softmax(dim=0)
     with torch.no_grad():
          fw = open("answer.txt", "w", encoding="utf-8")
          with open("Predict/labels_part"+str(part)+".txt", "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in tqdm(lines):
                    file = "Predict/part"+str(part) +"/" + line
                    
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to("cuda")
                    x = get_mel(wav.unsqueeze(0), h)
                    predict = model_unseen(x)
                    predict = predict.squeeze()

                    # predict_prob = softmax(predict)
                    # if predict_prob[1]>0.6:
                    #      fw.write( line+", 5\n" )
                    #      continue

                    if int(torch.argmax(predict).detach().cpu().numpy()):
                         fw.write( line+", 5\n" )
                         continue

                    predict = model_algo(x)
                    predict = predict.squeeze()
                    fw.write( line+", "+str(int(torch.argmax(predict).detach().cpu().numpy()))+"\n" )

def predict_6class(part, model_path, h):
     model = Classifier().to("cuda")
     state_dict_g = torch.load(model_path, map_location="cuda")
     model.load_state_dict(state_dict_g['generator'])
     model.eval()

     data = MelDataset([], h)
     with torch.no_grad():
          fw = open("answer.txt", "w", encoding="utf-8")
          with open("Predict/labels_part"+str(part)+".txt", "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in tqdm(lines):
                    file = "Predict/part"+str(part) +"/" + line
                    
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to("cuda")
                    x = get_mel(wav.unsqueeze(0), h)
                    predict = model(x)
                    predict = predict.squeeze()
                    fw.write( line+", "+str(int(torch.argmax(predict).detach().cpu().numpy()))+"\n" )

def predict():

     parser = argparse.ArgumentParser()
     parser.add_argument('--checkpoint_path_unseen', default='')
     parser.add_argument('--checkpoint_path', default='')
     parser.add_argument('--config', default='config.json')
     parser.add_argument('--part')
     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     json_config = json.loads(data)
     h = AttrDict(json_config)
     torch.manual_seed(h.seed)

     if not a.checkpoint_path_unseen:
          predict_6class(a.part, model_path= a.checkpoint_path, h = h)
     else:
          predict_cascade(a.part, a.checkpoint_path_unseen, a.checkpoint_path, h)


if __name__ == '__main__':
     predict()
     # python predict.py --part 1 --checkpoint_path Outdir/ex1/g_00038000 --config Outdir/ex1/config.json   => Data part 1 voting 86 , nosegment 90.8
     # python predict.py --part 1 --checkpoint_path Outdir/ex2/g_00038000 --config Outdir/ex2/config.json   => Data part 1 voting 86 , nosegment 90.8
     # python predict.py --part 1 --checkpoint_path Outdir/ex3/g_00130000 --config Outdir/ex3/config.json => Data part 2(x5) nosegment part1=93
     # python predict.py --part 2 --checkpoint_path Outdir/ex3/g_00130000 --config Outdir/ex3/config.json => Data part 2(x5) nosegment part1=93.2 part2=93.2
     # python predict.py --part 2 --checkpoint_path_unseen Outdir/ex4/g_00040000  --checkpoint_path Outdir/ex5/g_00031000 --config Outdir/ex4/config.json # Cascade part2=85.9
     # python predict.py --part 2 --checkpoint_path_unseen Outdir/ex6/g_00026000  --checkpoint_path Outdir/ex5/g_00030000 --config Outdir/ex4/config.json # Cascade, hard augment spec, augment 5k old, use focal loss part2=93.97, 1081 class unseen
     # python predict.py --part 2 --checkpoint_path_unseen Outdir/ex8/g_00026000  --checkpoint_path Outdir/ex9/g_00080000 --config Outdir/ex8/config.json # hard augment, 30k files, 25k for training, increase batch, effi b1 for 5 classes # part2=94.3
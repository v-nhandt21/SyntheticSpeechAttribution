import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tudataset import TuDataset, get_dataset_filelist
from utils import save_checkpoint
import os
from utils import AttrDict, save_checkpoint, build_env
from tqdm import tqdm
from model import Tulet
import warnings, glob
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
from loss import ArcMarginProduct

# This require batch is divided by 4

def train(rank, a, h):

     model = Tulet(h=h).to("cuda")

     cp_list = glob.glob(a.checkpoint_path+"/"+"g_*")
     if len(cp_list) != 0:
          model.load_state_dict(torch.load(sorted(cp_list)[-1], map_location="cuda")['generator'])

     steps = 0
     last_epoch = -1

     optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate)

     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

     training_filelist, validation_filelist = get_dataset_filelist(a)

     trainset = TuDataset(training_filelist, h)

     train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True, sampler=None, batch_size=h.batch_size, pin_memory=False, drop_last=True)

     if rank == 0:
          validset = TuDataset(validation_filelist, h)
          validation_loader = DataLoader(validset, num_workers=4, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)

          sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

     
     model.train()

     criterion = torch.nn.CrossEntropyLoss()

     train_err_tot = 0

     for epoch in tqdm(range(max(0, last_epoch), a.training_epochs)):
          for i, batch in enumerate(train_loader):
               ac, p, n, u = batch

               ac = torch.autograd.Variable(ac.to("cuda", non_blocking=True))
               p = torch.autograd.Variable(p.to("cuda", non_blocking=True))
               n = torch.autograd.Variable(n.to("cuda", non_blocking=True))
               u = torch.autograd.Variable(u.to("cuda", non_blocking=True))
               
               ac_h = model(ac)
               p_h = model(p)
               n_h = model(n)
               u_h = model(u)

               loss = criterion(ac_h, p_h) - criterion(ac_h, n_h) - criterion(ac_h, u_h) - criterion(n_h, u_h)
               
               train_err_tot += loss
               optim_g.zero_grad()
               loss.backward()
               optim_g.step()

               if rank == 0:
                    if steps % 1000 == 0 and steps != 0:
                         checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                         save_checkpoint(checkpoint_path, {'generator': (model).state_dict()})

                    # Tensorboard summary logging
                    if steps % 100 == 0:
                         sw.add_scalar("training/gen_loss_total", 10+train_err_tot/100, steps)
                         sw.add_scalar("lr",scheduler_g.get_last_lr()[-1],steps)
                         train_err_tot = 0

                    # Validation
                    if steps % 200 == 0:  # and steps != 0:
                         model.eval()
                         torch.cuda.empty_cache()
                         val_err_tot = 0
                         with torch.no_grad():
                              for j, batch in enumerate(validation_loader):
                                   ac, p, n, u = batch
                                   ac = torch.autograd.Variable(ac.to("cuda", non_blocking=True))
                                   p = torch.autograd.Variable(p.to("cuda", non_blocking=True))
                                   n = torch.autograd.Variable(n.to("cuda", non_blocking=True))
                                   u = torch.autograd.Variable(u.to("cuda", non_blocking=True))
                                   
                                   ac_h = model(ac)
                                   p_h = model(p)
                                   n_h = model(n)
                                   u_h = model(u)

                                   loss = criterion(ac_h, p_h) - criterion(ac_h, n_h) - criterion(ac_h, u_h) - criterion(n_h, u_h)
                                   val_err_tot += loss

                         sw.add_scalar("validation/loss", 10+val_err_tot/(j+1), steps)
                         model.train()
               steps += 1
          scheduler_g.step()

def main():

     parser = argparse.ArgumentParser()
     parser.add_argument('--input_training_file', default='')
     parser.add_argument('--input_validation_file', default='')
     parser.add_argument('--checkpoint_path', default='')
     parser.add_argument('--config', default='')
     parser.add_argument('--training_epochs', default=250, type=int)

     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     json_config = json.loads(data)
     h = AttrDict(json_config)
     build_env(a.config, 'config.json', a.checkpoint_path)
     os.makedirs(a.checkpoint_path, exist_ok=True)
     torch.manual_seed(h.seed)
     train(0, a, h)


if __name__ == '__main__':
     main()
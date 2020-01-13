import os
import time
import torch
import json
from glob import glob


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Test Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['test_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))

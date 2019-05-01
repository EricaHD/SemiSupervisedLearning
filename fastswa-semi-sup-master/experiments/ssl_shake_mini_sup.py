import sys
import logging
import torch
sys.path.append('.')
import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
     
    defaults = {
        # Technical details
        'workers': 4,
        'checkpoint_epochs': 10,
        'print_freq':100,

        # Data
        'dataset': 'sslMini',
        'train_subdir': 'supervised/train',
        'eval_subdir': 'supervised/val',
        'exclude_unlabeled':True,
        
        # Architecture
        'arch': 'cifar_shakeshake26',
        'ema_decay': 0.97,

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 1000,
        'logit_distance_cost': -1,
        'weight_decay': 2e-4,

        # Optimization
        'epochs': 100,
        'lr_rampup': 0,
        'lr': 0.1,
        'lr_rampdown_epochs': 150,
        'nesterov': True,

        'num_cycles': 20,
        'cycle_interval': 30,
        'start_epoch': 0,
        'fastswa_frequencies': '3',
        
        'device':'cuda',
        'title' : 'ssl_shake_mini_sup',
        'data_seed':10, 
        
        'batch_size': 256
        
    }
    
    return defaults

def run(title, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    context = RunContext('/scratch/ijh216/ssl/', __file__, "{}".format(data_seed))
    main.args = parse_dict_args(**kwargs)
    main.main(context)


if __name__ == "__main__":
    run(**parameters())

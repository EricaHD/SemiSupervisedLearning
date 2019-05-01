import sys
import logging
import torch
sys.path.append('.')
import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
    
    ngpu = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    base_lr = 0.1
    base_batch_size = 256
    base_labeled_batch_size = 65
    
    defaults = {
        # Technical details
        'workers': 3,
        'checkpoint_epochs': 3,

        # Data
        'dataset': 'ssl',
        'train_subdir': 'supervised/train',
        'unsup_subdir': 'unsupervised',
        'eval_subdir': 'supervised/val',

        # Data sampling
        'base_batch_size': 256,
        'base_labeled_batch_size': 56,

        # Architecture
        'arch': 'cifar_shakeshake26',
        'ema_decay': 0.97,

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'epochs': 8,
        'lr_rampup': 0,
        'base_lr': 0.1,
        'lr_rampdown_epochs': 10,
        'nesterov': True,

        'num_cycles': 20,
        'cycle_interval': 30,
        'start_epoch': 0,
        'fastswa_frequencies': '3',
        
        'device':'cuda',
        'title' : 'ssl_mt_shake_short',
        'data_seed':10, 
        
        'batch_size': base_batch_size * ngpu, 
        'labeled_batch_size': base_labeled_batch_size * ngpu, 
        'lr': base_lr * ngpu
        
    }
    
    return defaults

def run(title, base_batch_size, base_labeled_batch_size, base_lr, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    context = RunContext('/scratch/ijh216/ssl/', __file__, "{}".format(data_seed))
    main.args = parse_dict_args(**kwargs)
    main.main(context)


if __name__ == "__main__":
    run(**parameters())

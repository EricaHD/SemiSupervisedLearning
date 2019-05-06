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
        'checkpoint_epochs': 5,
        'evaluation_epochs':5,
        'resume':'/scratch/ijh216/ssl/ssl_shake_mini/2019-05-01_19-04-25/10/transient/checkpoint.230.ckpt',

        # Architecture
        'arch': 'cifar_shakeshake26',
        'ema_decay': 0.999,

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 155,
        'consistency': 500000.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'epochs': 200,
        'augment_unlabeled_epoch':150,
        'start_epoch':145
        'lr': 0.1,
        'lr_rampup': 0,
        'lr_rampdown_epochs': 230,
        'nesterov': True,

        'num_cycles': 20,
        'cycle_interval': 30,
        'start_epoch': 0,
        'fastswa_frequencies': '3',
        
        'device':'cuda',
        'title' : 'ssl_shake_mini_augment',
        'data_seed':10, 
        
        'batch_size': 256,
        'labeled_batch_size': 65
        
        
    }
    
    return defaults

def run(title, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    context = RunContext('/scratch/ijh216/ssl/', __file__, "{}".format(data_seed))
    main.args = parse_dict_args(**kwargs)
    main.main(context)


if __name__ == "__main__":
    run(**parameters())
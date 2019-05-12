import sys
import logging
import torch
sys.path.append('.')
import main_2 as main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
    
    ngpu = torch.cuda.device_count()
    
    defaults = {
        # Technical details
        'workers': 4,
        'checkpoint_epochs': 25,
        'evaluation_epochs':5,
        
        # Data
        'dataset': 'sslK',
        'train_subdir': 'train_16',
        'unsup_subdir': 'unsupervised',
        'eval_subdir': 'supervised/val',
        'augment_unlabeled_init':True,
        'augment_unlabeled_epoch':-1,

        # Architecture
        'arch': 'cifar_shakeshake26',
        'ema_decay': 0.999,

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 150,
        'consistency': 1000.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,
        'innovate':False,

        # Optimization
        'epochs': 50,
        'start_epoch':0,
        'lr': 0.1 * ngpu,
        'lr_rampup': 0,
        'lr_rampdown_epochs': 75,
        'nesterov': True,

        'num_cycles': 3,
        'cycle_interval': 3,
        'fastswa_frequencies': '3',
        
        'device':'cuda',
        'title' : 'sslK_16',
        'data_seed':10, 
        
        'batch_size': 384 * ngpu,
        'labeled_batch_size': 153 * ngpu
        
        
    }
    
    return defaults

def run(title, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    context = RunContext('/scratch/ijh216/ssl/', __file__, "{}".format(data_seed))
    main.args = parse_dict_args(**kwargs)
    main.main(context)


if __name__ == "__main__":
    run(**parameters())

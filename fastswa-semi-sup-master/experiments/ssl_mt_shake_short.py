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
        'workers': 2,
        'checkpoint_epochs': 3,

        # Data
        'dataset': 'ssl',
        'train_subdir': 'supervised/train',
        'unsup_subdir': 'unsupervised',
        'eval_subdir': 'supervised/val',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

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
        'epochs': 180,
        'lr_rampup': 0,
        'base_lr': 0.1,
        'lr_rampdown_epochs': 210,
        'nesterov': True,

        'num_cycles': 20,
        'cycle_interval': 30,
        'start_epoch': 0,
        'fastswa_frequencies': '3'
    }
        
    yield {
           **defaults,
           'title': 'ssl_mt_shake_short',
           'data_seed': 10
          }



def run(title, base_batch_size, base_labeled_batch_size, base_lr, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    ngpu = torch.cuda.device_count()
    adapted_args = {
        'batch_size': base_batch_size * 1,
        'labeled_batch_size': base_labeled_batch_size * 1,
        'lr': base_lr * ngpu,
    }
    context = RunContext('/scratch/ijh216/ssl/', __file__, "{}".format(data_seed))
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main(context)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)

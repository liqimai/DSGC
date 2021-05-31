from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config
import argparse
import pprint

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", help='print more info')
parser.add_argument("--seed", type=int, help='Random seed.')
parser.add_argument("--repeat", type=int, help='the number of runs to average results')

dataset_arguments = parser.add_argument_group('dataset arguments')
dataset_arguments.add_argument("--dataset", type=str, help='dataset, e.g., large_cora, 20news or wiki')
dataset_arguments.add_argument("--train-size", type=str, help='number of labels per class for train')
dataset_arguments.add_argument("--valid-size", type=str, help='number of labels per class for validation')
dataset_arguments.add_argument("--validate", action="store_true", help='use validation set. By default, it is not used.')
dataset_arguments.add_argument("--no-validate", action="store_true", help='don\'t use validation set. By default, it is not used.')


training_arguments = parser.add_argument_group('training arguments')
training_arguments.add_argument("--epochs", type=int, help='training epochs')
training_arguments.add_argument("--learning-rate", type=float, help='learning rate')
training_arguments.add_argument("--dropout", type=float, help='dropout probability, from 0.0 to 1.0')
training_arguments.add_argument("--weight-decay", type=float, help='L2 regularization')
training_arguments.add_argument("--layer-size", type=eval, help='a python list of hidden layer widths')

predefined_models = parser.add_argument_group('predefined models')
predefined_models.add_argument("--X", action="store_true", help='predifined baselines, LP, MLP')
predefined_models.add_argument("--GCN", action="store_true", help='GCN with 2D Conv')
predefined_models.add_argument("--GLP", action="store_true", help='GLP with 2D Conv')
predefined_models.add_argument("--XF", action="store_true", help='predifined model XF + MLP')
predefined_models.add_argument("--GXF", action="store_true", help='predifined model GXF + MLP')

args = parser.parse_args()
print(args)

configuration = {
    # repeating times
    'repeating'             : 1,

    # The default model configuration
    'default':{
        # dataset
        'dataset'           : '20news',   # 'data/large_cora.mat'
        'shuffle'           : True,
        'train_size'        : 20,         # if train_size is a number, then use TRAIN_SIZE labels per class.
        'validation_size'   : 500,          # 'Use VALIDATION_SIZE data to train model'
        'validate'          : True,       # Whether use validation set
        'test_size'         : None,       # If None, all rest are test set

        # Model
        'Model': 'MLP',  # 'GCN'/ 'LP' / 'MLP' / 'DSGC' / 'GLP'
        'G': None,
        'F': None,

        'learning_rate'     : 0.1,        # 'Initial learning rate.'
        'epochs'            : 200,        # 'Number of epochs to train.'
        'dropout'           : 0.2,                # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,             # 'Weight for L2 loss on embedding matrix.'

        'connection'        : 'ff',
        # A string contains only char "c" or "f".
        # "c" stands for convolution.
        # "f" stands for fully connected.
        # See layer_size for details.

        'layer_size'        : [64],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        'cvr': None,

        'random_seed'       : int(time.time()), #'Random seed.'

        'logging'           : False,            # 'Weather or not to record log'
        'logdir'            : 'model/',         # 'Log directory.''
        'model_dir'         : 'model/',
        'name'              : None,             # 'name of the model. Serve as an ID of model.'

        # 'threads'           : 2*cpu_count(),    #'Number of threads'
        'train'             : True,
        'inter-intra-var'   : False,
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':
    [
        # {
        #     'Model'     : 'DSGC',
        #     'G': True if not args.dataset.startswith('webkb') else 'LP',
        #     'F': F,
        # }
        # for F in [ None, 'Emb_sym', 'PMI_sym']
    ]
}


if args.verbose is not None:
    configuration['default']['verbose'] = args.verbose
if args.seed is not None:
    configuration['default']['random_seed']=args.seed
if args.repeat is not None:
    configuration['repeating']=args.repeat

if args.dataset is not None:
    configuration['default']['dataset'] = args.dataset
if args.train_size is not None:
    configuration['default']['train_size'] = eval(args.train_size)
if args.valid_size is not None:
    configuration['default']['validation_size'] = eval(args.valid_size)

assert not (args.validate and args.no_validate), "argument --validate: not allowed with argument --no-validate"
if args.validate:
    configuration['default']['validate'] = True
if args.no_validate:
    configuration['default']['validate'] = False

if args.epochs is not None:
    configuration['default']['epochs'] = args.epochs
if args.learning_rate is not None:
    configuration['default']['learning_rate'] = args.learning_rate
if args.dropout is not None:
    configuration['default']['dropout'] = args.dropout
if args.weight_decay is not None:
    configuration['default']['weight_decay'] = args.weight_decay
if args.layer_size is not None:
    configuration['default']['layer_size'] = args.layer_size

if args.X:
    configuration['model_list'] += [
        # LP
        {
            'Model' : 'LP',
            'alpha' : 100,
        },
        # MLP
        {
            'Model'     : 'MLP',
        },
    ]
if args.GCN:
    configuration['model_list'] += [
        # GCN
        {
            'Model'     : 'GCN',
            'connection': 'cc',
            'F'         : F,
        }
        for F in [None, 'Emb_sym', 'PMI_sym']
    ]
if args.GLP:
    configuration['model_list'] += [
        # GLP
        {
            'Model'     : 'GLP',
            'G'         : 'LP',
            'F'         : F,
        }
        for F in [None, 'Emb_sym', 'PMI_sym']
    ]
if args.XF:
    configuration['model_list'] +=  [
        {
            'Model'     : 'DSGC',
            'G': False,
            'F': F,
        }
        for F in [ None, 'Emb_sym', 'PMI_sym']
    ]
if args.GXF:
    configuration['model_list'] +=  [
        {
            'Model'     : 'DSGC',
            'G': True if not args.dataset.startswith('webkb') else 'LP',
            'F': F,
        }
        for F in [ None, 'Emb_sym', 'PMI_sym']
    ]

pprint.PrettyPrinter(indent=4).pprint(configuration)
# exit()

def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

configuration['model_list'] = list(map(set_default_attr,
    configuration['model_list']))

for model_config in configuration['model_list']:
    preprocess_model_config(model_config)


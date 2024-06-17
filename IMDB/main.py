import torch
import torch.nn as nn
import torch.backends.cudnn 
# import torch.utils.tensorboard
import torch.utils.data

import os
import json 
import random

import train
import valid

import model.get_model
from transformers import BertTokenizer
from torchtext.legacy import data, datasets

import utils.option

import warmup_scheduler

args = utils.option.get_args_parser()
torch.backends.cudnn.benchmark = True

if args.attn_type == 'softmax':
    save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
elif args.attn_type == 'kep_svgp':
    save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# writer = torch.utils.tensorboard.SummaryWriter(save_path)
writer = None
logger = utils.utils.get_logger(save_path)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

## Load the IMDB dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens
    
TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = tokenizer.cls_token_id,
                  eos_token = tokenizer.sep_token_id,
                  pad_token = tokenizer.pad_token_id,
                  unk_token = tokenizer.unk_token_id)

LABEL = data.LabelField(dtype = torch.float)

traind, testd = datasets.IMDB.splits(TEXT, LABEL)
traind.examples = traind.examples + testd.examples 
all_data = traind 
train_data, test_data = all_data.split(random_state = random.seed(args.seed), split_ratio=0.8) 
train_data, valid_data = train_data.split(random_state = random.seed(args.seed), split_ratio=0.875)

LABEL.build_vocab(train_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = args.batch_size, 
    device = device)

for run in range(args.nb_run):
    prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
    logger.info(100*'#' + '\n' + prefix)

    ## define model
    net = model.get_model.get_model(args.model, len(tokenizer.vocab), logger, args)
    net.cuda()
    ## define optimizer with warm-up
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
    
    ## make logger
    best_acc, best_auroc, best_aurc = 0, 0, 1e6

    ## start training
    for epoch in range(args.nb_epochs):
        train.train(train_iterator, net, optimizer, epoch, logger, writer, args)
        
        scheduler.step()

        # validation
        net_val = net
        res = valid.validation(valid_iterator, net_val, args) 
        log = [key + ': {:.3f}'.format(res[key]) for key in res]
        msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
        logger.info(msg)

        # for key in res :
        #     if run < 1:
        #         writer.add_scalar('./Val/' + key, res[key], epoch)

        if res['Acc.'] > best_acc :
            acc = res['Acc.']
            msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
            logger.info(msg)
            best_acc = acc
            torch.save(net_val.state_dict(),os.path.join(save_path, f'best_acc_net_{run+1}.pth'))
        
        if res['AUROC'] > best_auroc :
            auroc = res['AUROC']
            msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
            logger.info(msg)
            best_auroc = auroc
            torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
    
        if res['AURC'] < best_aurc :
            aurc = res['AURC']
            msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
            logger.info(msg)
            best_aurc = aurc
            torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))

def process_results(args, loader, model, metrics, logger, method_name, results_storage):
    res = valid.validation(loader, model, args)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
results_storage = {metric: [] for metric in metrics}

for r in range(args.nb_run):
    logger.info(f'Testing model_{r + 1} ...')

    net = model.get_model.get_model(args.model, len(tokenizer.vocab), logger, args)
    net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
    net = net.cuda()
    process_results(args, test_iterator, net, metrics, logger, "MSP", results_storage)

results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
test_results_path = os.path.join(save_path, 'test_results_imdb.csv')
utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
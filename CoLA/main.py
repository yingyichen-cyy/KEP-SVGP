import torch
import torch.nn as nn
import torch.backends.cudnn 
import torch.utils.tensorboard

import os
import json 

import train
import valid

import model.get_model
from data_loader import get_data, get_vocab, DataLoader
import utils.option

import warmup_scheduler


args = utils.option.get_args_parser()
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.attn_type == 'softmax':
    save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
elif args.attn_type == 'kep_svgp':
    save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)

if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = torch.utils.tensorboard.SummaryWriter(save_path)
logger = utils.utils.get_logger(save_path)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

data_train,gold_train,data_test,gold_test,data_ood,gold_ood=\
        get_data(['./data/in_domain_train.tsv','./data/in_domain_dev.tsv'],['./data/out_of_domain_dev.tsv'], args.seed)
word_to_int, _ = get_vocab(data_train, args.min_word_count)
vocab_size = len(word_to_int)

train_loader = DataLoader(data_train,gold_train,args.batch_size,word_to_int,device)
test_loader = DataLoader(data_test,gold_test,args.batch_size,word_to_int,device,shuffle=False)

for run in range(args.nb_run):
    prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
    logger.info(100*'#' + '\n' + prefix)

    ## define model
    net = model.get_model.get_model(args.model, vocab_size, logger, args)
    net.cuda()
    ## define optimizer with warm-up
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
    
    ## make logger
    best_mcc, best_auroc, best_aurc = 0, 0, 1e6

    ## start training
    for epoch in range(args.nb_epochs):
        train.train(train_loader, net, optimizer, epoch, logger, writer, args)
        
        scheduler.step()

        # validation
        net_val = net
        res = valid.validation(test_loader, net_val, args) 
        log = [key + ': {:.3f}'.format(res[key]) for key in res]
        msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
        logger.info(msg)

        for key in res :
            if run < 1:
                writer.add_scalar('./Val/' + key, res[key], epoch)

        if res['MCC'] > best_mcc :
            mcc = res['MCC']
            msg = f'MCC improved from {best_mcc:.2f} to {mcc:.2f}!!!'
            logger.info(msg)
            best_mcc = mcc
            torch.save(net_val.state_dict(),os.path.join(save_path, f'best_mcc_net_{run+1}.pth'))
        
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
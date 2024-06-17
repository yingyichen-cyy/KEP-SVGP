import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import utils.utils


def compute_loss(cls_criterion, preds, targets, score_list=None, lambda_sqrt_inv_list=None, kl_list=None, eta_ksvd=1., eta_kl=1.):
    loss_ce = cls_criterion(preds, targets)

    if score_list is not None:
        loss_ksvd = 0
        loss_kl = 0
        for i in range(len(score_list)):
            # loss_ksvd
            loss_escore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][0], lambda_sqrt_inv_list[i].unsqueeze(0))).norm(dim=-1, p=2)**2)/2
            loss_rscore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][1], lambda_sqrt_inv_list[i].unsqueeze(0))).norm(dim=-1, p=2)**2)/2
            loss_trace = torch.einsum('...ps,...pd->...sd', score_list[i][2], score_list[i][3].type_as(score_list[i][2])).mean(dim=0).trace()
            loss_ksvd = loss_ksvd + (loss_escore + loss_rscore - loss_trace) ** 2
            # loss_kl
            loss_kl = loss_kl + kl_list[i]

        # add loss_kl
        loss_total = loss_ce + eta_ksvd * loss_ksvd + eta_kl * loss_kl 

        return loss_total, loss_ce, loss_ksvd, loss_kl
    else:
        return loss_ce

def train(train_loader, net, optimizer, epoch, logger, writer, args):

    net.train()

    # define criterion
    cls_criterion = nn.CrossEntropyLoss()

    if args.attn_type == "softmax":
        train_log = {
            'MCC' : utils.utils.AverageMeter(),
            'Top1 Acc.' : utils.utils.AverageMeter(),
            'Tot. Loss' : utils.utils.AverageMeter(),
            'LR' : utils.utils.AverageMeter(),
            }
    elif args.attn_type == "kep_svgp":
        train_log = {
            'MCC' : utils.utils.AverageMeter(),
            'Top1 Acc.' : utils.utils.AverageMeter(),
            'CE Loss' : utils.utils.AverageMeter(),
            'KSVD Loss' : utils.utils.AverageMeter(),
            'KL Loss' : utils.utils.AverageMeter(),
            'Tot. Loss' : utils.utils.AverageMeter(),
            'LR' : utils.utils.AverageMeter(),
            }

    msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
    logger.info(msg)

    for i in range(train_loader.num_batches):
        data, inputs, inputs_mask, positional, answers = train_loader.__load_next__()
        inputs = inputs.cuda()
        inputs_mask = inputs_mask.cuda()
        positional = positional.cuda()
        answers = answers.cuda()

        optimizer.zero_grad()
        outs = net(inputs, positional, inputs_mask, data)

        if args.attn_type == "softmax":
            loss = compute_loss(cls_criterion, outs, answers)
        elif args.attn_type == "kep_svgp":
            loss, loss_ce, loss_ksvd, loss_kl = compute_loss(cls_criterion, outs[0], answers, \
                                                            outs[1], outs[2], outs[3], args.eta_ksvd, args.eta_kl)

        loss.backward()
        optimizer.step()

        if args.attn_type == "softmax":
            prec, _ = utils.utils.accuracy(outs, answers)
            mcc = utils.utils.mcc(outs, answers)
        elif args.attn_type == "kep_svgp":
            prec, _ = utils.utils.accuracy(outs[0], answers)
            mcc = utils.utils.mcc(outs[0], answers)

        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

        train_log['MCC'].update(mcc, inputs.size(0))
        train_log['Tot. Loss'].update(loss.item(), inputs.size(0))
        train_log['Top1 Acc.'].update(prec.item(), inputs.size(0))
        train_log['LR'].update(lr, inputs.size(0))
        if args.attn_type == "kep_svgp":
            train_log['CE Loss'].update(loss_ce.item(), inputs.size(0))
            train_log['KSVD Loss'].update(loss_ksvd.item(), inputs.size(0))
            train_log['KL Loss'].update(loss_kl.item(), inputs.size(0))

        if i % 100 == 99:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [key + ': {:.2f}'.format(train_log[key].avg) for key in train_log if key != 'LR']
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
            logger.info(msg)
            for key in train_log : 
                train_log[key] = utils.utils.AverageMeter()

    for key in train_log : 
        writer.add_scalar('./Train/' + key, train_log[key].avg, epoch)
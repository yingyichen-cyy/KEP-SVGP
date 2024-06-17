import torch
import torch.nn.functional as F
import utils.metrics
import numpy as np 
from sklearn.metrics import matthews_corrcoef


@torch.no_grad()
def validation(loader, net, args):
    net.eval()
    
    mcc_list = []
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}

    for i in range(loader.num_batches):
        data, inputs, inputs_mask, positional, answers = loader.__load_next__()
        inputs = inputs.cuda()
        inputs_mask = inputs_mask.cuda()
        positional = positional.cuda()
        answers = answers.cuda()

        if args.attn_type == "softmax":
            output = net(inputs, positional, inputs_mask, data)
            
        elif args.attn_type == "kep_svgp":
            results = []
            for _ in range(10):
                results.append(net(inputs, positional, inputs_mask, data)[0])
            outputs = torch.stack(results)
            output = torch.mean(outputs, 0)
            
        softmax = F.softmax(output, dim=1)
        _, pred_cls = softmax.max(1)

        val_log['correct'].append(pred_cls.cpu().eq(answers.cpu().data.view_as(pred_cls)).numpy())
        val_log['softmax'].append(softmax.cpu().data.numpy())
        val_log['logit'].append(output.cpu().data.numpy())
        val_log['target'].append(answers.cpu().data.numpy())

        mcc_list.append(matthews_corrcoef(answers.cpu().numpy(), pred_cls.detach().cpu().numpy()))
        
    for key in val_log : 
        val_log[key] = np.concatenate(val_log[key])
    ## acc
    acc = 100. * val_log['correct'].mean()
    ## mcc
    mcc = 100. * np.array(mcc_list).mean()

    # aurc, eaurc
    aurc, eaurc = utils.metrics.calc_aurc_eaurc(val_log['softmax'], val_log['correct'])
    # fpr, aupr
    auroc, aupr_success, aupr, fpr = utils.metrics.calc_fpr_aupr(val_log['softmax'], val_log['correct'])
    # calibration measure ece , mce, rmsce
    ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
    # brier, nll
    nll, brier = utils.metrics.calc_nll_brier(val_log['softmax'], val_log['logit'], val_log['target'])

    # log
    res = {
        'Acc.': acc,
        'MCC': mcc,
        'FPR' : fpr*100,
        'AUROC': auroc*100,
        'AUPR': aupr*100,
        'AURC': aurc*1000,
        'EAURC': eaurc*1000,
        'AUPR Succ.': aupr_success*100,
        'ECE' : ece*100,
        'NLL' : nll*10,
        'Brier' : brier*100
    }

    return res
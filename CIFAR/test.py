import torch
import valid
import os
import utils.test_option
import data.dataset
import data.CIFARC
import utils.utils
import model.get_model
import csv
from torch.utils.data import DataLoader
import torchvision.transforms

def process_results(args, loader, model, metrics, logger, method_name, results_storage):
    res = valid.validation(loader, model, args)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

def test_cifar_c_corruptions(dataset, model, corruption_dir, transform_test, batch_size, metrics, logger):
    if dataset == "cifar10":
        cor_results_storage = {corruption: {severity: {metric: [] for metric in metrics} for severity in range(1, 6)} for
                           corruption in data.CIFARC.CIFAR10C.cifarc_subsets}
        for corruption in data.CIFARC.CIFAR10C.cifarc_subsets:
            for severity in range(1, 6):
                logger.info(f"Testing on corruption: {corruption}, severity: {severity}")
                corrupted_test_dataset = data.CIFARC.CIFAR10C(root=corruption_dir, transform=transform_test, subset=corruption,
                                                            severity=severity, download=True)
                corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=4, drop_last=False)
                res = valid.validation(corrupted_test_loader, model, args)
                for metric in metrics:
                    cor_results_storage[corruption][severity][metric].append(res[metric])

    return cor_results_storage

def test():

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    cor_results_all_models = {}

    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)
    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        _, valid_loader, test_loader, nb_cls = data.dataset.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        print(nb_cls)
        net = model.get_model.get_model(args.model, nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        net = net.cuda()
        process_results(args, test_loader, net, metrics, logger, "MSP", results_storage)

        if args.dataset == 'cifar10':
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            cor_results_storage = test_cifar_c_corruptions(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
                                                            metrics, logger)
            cor_results = {corruption: {
                severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
                in range(1, 6)} for corruption in data.CIFARC.CIFAR10C.cifarc_subsets}
            cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    test_results_path = os.path.join(save_path, 'test_results.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    if args.dataset == 'cifar10':
        utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)

if __name__ == '__main__':
    args = utils.test_option.get_args_parser()
    test()
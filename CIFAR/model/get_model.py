import model.vit_cifar


def get_model(model_name, nb_cls, logger, args):
  
    if model_name == "vit_cifar":
        net = model.vit_cifar.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
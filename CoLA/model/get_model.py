import model.vit_cola


def get_model(model_name, vocab_size, logger, args):
  
    if model_name == "vit_cola":
        net = model.vit_cola.vit_cola(args=args, vocab_size=vocab_size, attn_type=args.attn_type, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
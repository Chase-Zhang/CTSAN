def set_template(args):
    if args.template == 'CTSAN':
        args.task = "VideoDeblur"
        args.model = "CTSAN"
        args.n_sequence = 5
        args.n_frames_per_video = 8000
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 80
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))

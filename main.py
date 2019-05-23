from __future__ import print_function
import argparse
import config as cfg


def program_config(parser):
    parser.add_argument('--run_model', default=cfg.run_model, type=str)
    parser.add_argument('--model_type', default=cfg.model_type, type=str)
    parser.add_argument('--mle_epoch', default=cfg.MLE_train_epoch, type=int)
    parser.add_argument('--adv_epoch', default=cfg.ADV_train_epoch, type=int)
    # parser.add_argument('--inter_epoch', default=cfg.inter_epoch, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--adv_g_step', default=cfg.ADV_g_step, type=int)
    # parser.add_argument('--rollout_num', default=cfg.rollout_num, type=int)
    # parser.add_argument('--d_step', default=cfg.d_step, type=int)
    # parser.add_argument('--d_epoch', default=cfg.d_epoch, type=int)
    parser.add_argument('--adv_d_step', default=cfg.ADV_d_step, type=int)
    # parser.add_argument('--adv_d_epoch', default=cfg.ADV_d_epoch, type=int)
    parser.add_argument('--gen_lr', default=cfg.gen_lr, type=float)
    parser.add_argument('--gen_adv_lr', default=cfg.gen_adv_lr, type=float)
    parser.add_argument('--dis_lr', default=cfg.dis_lr, type=float)
    parser.add_argument('--temp_adpt', default=cfg.temp_adpt, type=str)
    parser.add_argument('--temperature', default=cfg.temperature, type=int)
    parser.add_argument('--clip_norm', default=cfg.clip_norm, type=int)

    parser.add_argument('--cuda', default=cfg.CUDA, type=int)
    parser.add_argument('--device', default=cfg.device, type=int)
    parser.add_argument('--shuffle', default=cfg.data_shuffle, type=int)
    parser.add_argument('--ora_pretrain', default=cfg.oracle_pretrain, type=int)
    parser.add_argument('--gen_pretrain', default=cfg.gen_pretrain, type=int)
    parser.add_argument('--dis_pretrain', default=cfg.dis_pretrain, type=int)
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--save_root', default=cfg.save_root, type=str)
    parser.add_argument('--tips', default=cfg.tips, type=str)

    return parser


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()
    cfg.init_param(opt)

    # ==========Dict==========
    if cfg.if_real_data:
        from instructor.real_data.leakgan_instructor import LeakGANInstructor
        from instructor.real_data.seqgan_instructor import SeqGANInstructor
        from instructor.real_data.relgan_instructor import RelGANInstructor
    else:
        from instructor.oracle_data.leakgan_instructor import LeakGANInstructor
        from instructor.oracle_data.seqgan_instructor import SeqGANInstructor
        from instructor.oracle_data.relgan_instructor import RelGANInstructor
    instruction_dict = {
        'leakgan': LeakGANInstructor,
        'seqgan': SeqGANInstructor,
        'relgan': RelGANInstructor,
    }

    inst = instruction_dict[cfg.run_model](opt)
    if not cfg.if_test:
        inst._run()
    else:
        inst._test()
    try:
        inst.log.close()
    except:
        pass

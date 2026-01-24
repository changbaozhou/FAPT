import argparse
import copy
import torch

import os

from dass.utils import setup_logger, set_random_seed, collect_env_info
from dass.config import get_cfg_default
from dass.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.fapt
import trainers.zsclip

from mrfi import MRFI, EasyConfig

import pandas as pd


# prec = 'fp16'
prec = 'fp32'


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.FA = CN()
    cfg.TRAINER.FA.N_CTX = 32  # number of context vectors
    cfg.TRAINER.FA.CSC = False  # class-specific context
    cfg.TRAINER.FA.CTX_INIT = ""  # initialization words
    cfg.TRAINER.FA.PREC = prec  # fp16, fp32, amp
    cfg.TRAINER.FA.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE = 256
    cfg.DATASET.TRAIN_EPS = 8
    cfg.DATASET.TEST_EPS = 16
    # cfg.DATASET.NUM_SHOTS = 16



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # if not os.path.exists(args.path):
    #     os.makedirs(args.path)

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    # trainer = build_trainer(cfg)

    if args.eval_only:
        trainer = build_trainer(cfg)
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print(args.model_dir)
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        print('---------------------------------------------------')
        return
    
    if args.eval_fault:
        results = []
        trainer = build_trainer(cfg)
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print(args.model_dir)
        # print('---------------------------------------------------')
        # print('clean acc:')
        # trainer.test()
        # print('---------------------------------------------------')
        print('acc under fault:')
        # for ber in [1e-8, 1e-7, 1e-6, 1e-5]:
        # for ber in [1e-6, 1e-5]:


        
        # component-wise fault injection

        if args.fi_component_wise:


            for ber in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
            # for ber in [1e-9]:

                image_encoder = trainer.clip_model.visual
                text_encoder = trainer.clip_model.transformer
                econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')

                econfig.faultinject[0]['selector']['rate'] = ber
                if args.fi_image_encoder:
                    fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                    trainer.fi_image_encoder = fi_image_encoder
                    output_name = 'image_encoder'
                if args.fi_text_encoder:
                    fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                    trainer.fi_text_encoder = fi_text_encoder
                    output_name = 'text_encoder'
                if args.fi_both:
                    fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                    fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                    trainer.fi_image_encoder = fi_image_encoder
                    trainer.fi_text_encoder = fi_text_encoder
                    output_name = 'both'
                
                print(f'bit error rate: {ber}')
                res = trainer.test_fa()
                results.append((ber, res))
                # save fi results to csv
                df = pd.DataFrame(results, columns=['ber', 'acc'])
                df.to_csv(f'resilience_component_wise_{output_name}.csv', index=False)



        # layer -wise fault injection
        if args.fi_layer_wise:

            # for ber in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
            for ber in [1e-5]:
                image_encoder = trainer.clip_model.visual
                text_encoder = trainer.clip_model.transformer
                econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')

                econfig.faultinject[0]['selector']['rate'] = ber


                if args.fi_image_encoder:
                    fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                    cfg = fi_image_encoder.get_weights_configs()


                    for layer in range(len(image_encoder.transformer.resblocks)+1):

                        cfg.enabled = False 
                        if layer == 0:
                            cfg[layer].enabled = True
                        else: 
                            cfg[layer*3-2].enabled = True
                            cfg[layer*3-1].enabled = True
                            cfg[layer*3].enabled = True

                        trainer.fi_image_encoder = fi_image_encoder
                    

                        print(f'bit error rate: {ber}, layer: {layer}')
                        res = trainer.test_fa()
                        results.append((ber, layer, res))
                        df = pd.DataFrame(results, columns=['ber', 'layer', 'acc'])
                        df.to_csv(f'resilience_layer_wise_image.csv', index=False)


                
                if args.fi_text_encoder:
                    fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                    cfg = fi_text_encoder.get_weights_configs()


                    for layer in range(len(text_encoder.resblocks)):

                        cfg.enabled = False 

                        cfg[layer*3].enabled = True
                        cfg[layer*3+1].enabled = True
                        cfg[layer*3+2].enabled = True

                        print(cfg.enabled)


                        trainer.fi_text_encoder = fi_text_encoder
                    

                        print(f'bit error rate: {ber}, layer: {layer}')
                        res = trainer.test_fa()
                        results.append((ber, layer, res))
                        df = pd.DataFrame(results, columns=['ber', 'layer', 'acc'])
                        df.to_csv(f'resilience_layer_wise_text.csv', index=False)

        # bit -wise fault injection
        if args.fi_bit_wise:
            
            # for ber in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
            for ber in [1e-5]:
                image_encoder = trainer.clip_model.visual
                text_encoder = trainer.clip_model.transformer
                econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')

                econfig.faultinject[0]['selector']['rate'] = ber


                if args.fi_image_encoder:
                    for bit in range(32):
                        econfig.set_error_mode(0, {'method':'FloatFixedBitFlip','bit':bit,'floattype': 'float32'})
                        fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                        trainer.fi_image_encoder = fi_image_encoder
                    

                        print(f'bit error rate: {ber}, bit: {bit}')
                        res = trainer.test_fa()
                        results.append((ber, bit, res))
                        df = pd.DataFrame(results, columns=['ber', 'bit', 'acc'])
                        df.to_csv(f'resilience_bit_wise_image.csv', index=False)
                
                if args.fi_text_encoder:
                    for bit in range(32):
                        econfig.set_error_mode(0, {'method':'FloatFixedBitFlip','bit':bit,'floattype': 'float32'})
                        fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                        trainer.fi_text_encoder = fi_text_encoder
                    

                        print(f'bit error rate: {ber}, bit: {bit}')
                        res = trainer.test_fa()
                        results.append((ber, bit, res))
                        df = pd.DataFrame(results, columns=['ber', 'bit', 'acc'])
                        df.to_csv(f'resilience_bit_wise_text.csv', index=False)







            
        return

    if args.fa_training:
        trainer = build_trainer(cfg)
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        print('---------------------------------------------------')
        print('begin fa prompt tuning:')

        # if args.fapt_component:


        # if args.fapt_layer:


        # if args.fapt_bit:




        image_encoder = trainer.model.image_encoder
        econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')
        # econfig.set_error_mode(0, {'method':'FloatFixedBitFlip','bit':26,'floattype': 'float32'})
        


        econfig.faultinject[0]['selector']['rate'] = 1e-5
        fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
        trainer.fi_image_encoder = fi_image_encoder
        trainer.train_fa()  
        print('---------------------------------------------------')
        print('clean acc after fa prompt tuning:')
        trainer.test()
        print('---------------------------------------------------')
        print('begin fault injection test:')
        for ber in [1e-7]:
            image_encoder = trainer.model.image_encoder
            econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')
            econfig.faultinject[0]['selector']['rate'] = ber
            fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
            trainer.fi_image_encoder = fi_image_encoder
            print(f'bit error rate: {ber}')
            trainer.test_fa()   

        return

    if args.training:
        trainer = build_trainer(cfg)
        trainer.train()

        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()    

        return

    
    # elif not args.no_train:
    #     trainer = build_trainer(cfg)
    #     trainer.train()
    #     print('---------------------------------------------------')
    #     print('clean acc:')
    #     trainer.test()      
    #     return 


        







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/bobzhou/dataset", help="path to dataset")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--fa_training", action="store_true")



    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="FAPT", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval_only", action="store_true", help="evaluation only")


    parser.add_argument("--eval_fault", action="store_true", help="evaluation under fault injection")
    parser.add_argument("--fi_component_wise", action="store_true", help="evaluation under fault injection")
    parser.add_argument("--fi_layer_wise", action="store_true", help="evaluation under fault injection")
    parser.add_argument("--fi_bit_wise", action="store_true", help="evaluation under fault injection")
    

    parser.add_argument("--fi_image_encoder", action="store_true", help="evaluation under fault injection in image encoder")
    parser.add_argument("--fi_text_encoder", action="store_true", help="evaluation under fault injection in text encoder")
    parser.add_argument("--fi_both", action="store_true", help="evaluation under fault injection in both")




    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    args = parser.parse_args()

    # threading.Thread(target=log_gpu_usage).start()

    main(args)

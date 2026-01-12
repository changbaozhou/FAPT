import argparse
import copy
import torch

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

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))


    if args.eval_only:
        trainer = build_trainer(cfg)
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print(args.model_dir)
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        return
    
    if args.eval_fault:
        trainer = build_trainer(cfg)
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print(args.model_dir)
        # print('---------------------------------------------------')
        # print('clean acc:')
        # trainer.test()
        print('---------------------------------------------------')
        print('acc under fault:')
        # for ber in [1e-8, 1e-7, 1e-6, 1e-5]:
        # for ber in [1e-6, 1e-5]:
        for ber in [9e-2]:
            image_encoder = trainer.clip_model.visual
            text_encoder = trainer.clip_model.transformer
            econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')  
            econfig.set_error_mode(0, {'method':'FloatFixedBitFlip','bit':19,'floattype': 'float32'})
            econfig.faultinject[0]['selector']['rate'] = ber
            if args.fi_image_encoder:
                fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                trainer.fi_image_encoder = fi_image_encoder
            if args.fi_text_encoder:
                fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                trainer.fi_text_encoder = fi_text_encoder
            if args.fi_both:
                fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
                fi_text_encoder = MRFI(copy.deepcopy(text_encoder), econfig)
                trainer.fi_image_encoder = fi_image_encoder
                trainer.fi_text_encoder = fi_text_encoder
            print(f'bit error rate: {ber}')
            trainer.test_fa()
        return

    if args.fa_training:
        # trainer = build_trainer(cfg)
        # print('---------------------------------------------------')
        # print('clean acc:')
        # trainer.test()
        # print('---------------------------------------------------')
        # print('begin prompt tuning:')
        # trainer.train()
        # print('---------------------------------------------------')
        # print('acc after prompt tuning:')
        # trainer.test()
        # print('---------------------------------------------------')
        # print('begin fault injection test:')
        # for ber in [1e-6, 1e-5]:
        #     image_encoder = trainer.model.image_encoder
        #     econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')  
        #     econfig.faultinject[0]['selector']['rate'] = ber
        #     fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
        #     trainer.model.fi_image_encoder = fi_image_encoder
        #     print(f'bit error rate: {ber}')
        #     trainer.test_fa()

        
        trainer = build_trainer(cfg)
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        print('---------------------------------------------------')
        print('begin fa prompt tuning:')
        image_encoder = trainer.model.image_encoder
        econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')  
        econfig.set_error_mode(0, {'method':'FloatFixedBitFlip','bit':26,'floattype': 'float32'})
        econfig.faultinject[0]['selector']['rate'] = 1e-7
        fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
        trainer.model.fi_image_encoder = fi_image_encoder
        trainer.train_fa()
        print('---------------------------------------------------')
        print('acc after fa prompt tuning:')
        trainer.test()
        print('---------------------------------------------------')
        print('begin fault injection test:')
        for ber in [1e-7]:
            image_encoder = trainer.model.image_encoder
            econfig = EasyConfig.load_file('easyconfigs/float_weight_fi.yaml')  
            econfig.faultinject[0]['selector']['rate'] = ber
            fi_image_encoder = MRFI(copy.deepcopy(image_encoder), econfig)
            trainer.model.fi_image_encoder = fi_image_encoder
            print(f'bit error rate: {ber}')
            trainer.test_fa()
        




        






    elif not args.no_train:
        trainer = build_trainer(cfg)
        trainer.train()
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/xuzhewen/bobzhou/dataset", help="path to dataset")
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
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--eval-fault", action="store_true", help="evaluation with fault")

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

    args.root= "/xuzhewen/bobzhou/dataset"
    # args.eval_only = True

    # ------------test CLIP with fault-----------------
    # args.trainer = "ZeroshotCLIP"
    # args.eval_fault = True
    # args.fi_image_encoder = True
    # args.fi_text_encoder = False
    # args.fi_both = False
    # -----------------------------

    # ------------FA prompt tuning----------------- 
    args.trainer = "FAPT"
    args.fa_training = True

    args.load_epoch = 100
    args.seed = 1
    args.dataset_config_file = "/xuzhewen/bobzhou/FAPT/configs/datasets/oxford_flowers.yaml"
    args.config_file = "/xuzhewen/bobzhou/FAPT/configs/trainers/FAPT/vit_l14.yaml"
    args.output_dir = "/xuzhewen/bobzhou/FAPT/output/oxford_flowers/FAPT/vit_l14/fa"
    args.model_dir = "/xuzhewen/bobzhou/FAPT/output/oxford_flowers/FAPT/vit_l14/fa"



    main(args)

import os.path as osp
import torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import copy
import time

from dass.engine import TRAINER_REGISTRY, TrainerX
from dass.metrics import compute_accuracy
from dass.utils import load_pretrained_weights, load_checkpoint
from dass.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from mrfi import MRFI, EasyConfig


_tokenizer = _Tokenizer()


# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
# }

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url, '/vhome/user/.cache/clip/')
    model_path = clip._download(url, '/home/bobzhou/.cache/clip')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FA.N_CTX
        ctx_init = cfg.TRAINER.FA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.FA.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FA.CLASS_TOKEN_POSITION


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class FACLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


        """MRFI"""
        self.fi_image_encoder = None
        self.fi_text_encoder = None





    def forward(self, image):
        """ original """
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits


    def forward_fa(self, image):
        """ faward with fault injection """

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts


        if self.fi_image_encoder is not None:
            image_features = self.fi_image_encoder(image.type(self.dtype))
        else:
            image_features = self.image_encoder(image.type(self.dtype))

        if self.fi_text_encoder is not None:
            x = prompts + self.fi_text_encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.fi_text_encoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.fi_text_encoder.text_projection
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits




@TRAINER_REGISTRY.register()
class FAPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.FA.PREC == "fp32" or cfg.TRAINER.FA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = FACLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FA.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    # def forward_backward_fa(self, batch):
    #     image, label = self.parse_batch_train(batch)

    #     prec = self.cfg.TRAINER.FA.PREC
    #     if prec == "amp":
    #         with autocast():
    #             output = self.model.forward_fa(image)
    #             loss = F.cross_entropy(output, label)
    #         self.optim.zero_grad()
    #         self.scaler.scale(loss).backward()
    #         self.scaler.step(self.optim)
    #         self.scaler.update()
    #     else:
    #         output = self.model.forward_fa(image)
    #         loss = F.cross_entropy(output, label)
    #         self.model_backward_and_update(loss)

    #     loss_summary = {
    #         "loss": loss.item(),
    #         "acc": compute_accuracy(output, label)[0].item(),
    #     }

    #     if (self.batch_idx + 1) == self.num_batches:
    #         self.update_lr()

    #     return loss_summary
    
    def forward_backward_fa(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FA.PREC
        if prec == "amp":
            with autocast():
                output_fa = self.model.forward_fa(image)
                loss_fa = F.cross_entropy(output_fa, label)

                # output_org = self.model(image)
                # loss_org = F.cross_entropy(output_org, label)

                # loss = loss_org + 0.1 * loss_fa
                loss = loss_fa 

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_fa = self.model.forward_fa(image)
            loss_fa = F.cross_entropy(output_fa, label)

            # output_org = self.model(image)
            # loss_org = F.cross_entropy(output_org, label)

            # loss = loss_org + 0.9 * loss_fa
            loss = loss_fa 

            output = output_fa

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def model_inference(self, image):

        logits = self.model(image)
        return logits
    
    def model_inference_fa(self, image):

        logits = self.model.forward_fa(image)
        return logits

import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dass.data import DataManager
from dass.optim import build_optimizer, build_lr_scheduler
from dass.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dass.modeling import build_head, build_backbone
from dass.evaluation import build_evaluator
# from attack.attackFeature import PGD
# from attack.purification import super_resolution
import clip
from torch import randperm
import os


def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
            self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False
        print(names)
        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()


        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def train_fa(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()


        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch_fa()
            self.after_epoch()
        self.after_train()


    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError
    
    def run_epoch_fa(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
    
    def test_fa(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_inference_fa(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class ClipModel(torch.nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ClipModel, self).__init__()
        self.model = model

        # temp, self.preprocess_val = clip.load(self.name, 'cpu')
        self.visual_encoder = self.model

        output_dim = self.visual_encoder.output_dim
        # add a fc for attacking embedding
        self.fc = torch.nn.Linear(output_dim, 2)

    def forward(self, image):
        x = self.visual_encoder(image)
        x = self.fc(x)
        return x


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        dm = DataManager(self.cfg, batch_size)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        # optional
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        dm = DataManager(self.cfg, batch_size)
        self.train_loader_x_notransform_noshuffle = dm.train_loader_x

        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE
        dm = DataManager(self.cfg, batch_size)
        self.train_loader_x_noshuffle = dm.train_loader_x

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def train_fa(self):
        super().train_fa(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()





    def after_train(self):
        print("Finish training")

        # do_test = not self.cfg.TEST.NO_TEST
        # if do_test:
        #     if self.cfg.TEST.FINAL_MODEL == "best_val":
        #         print("Deploy the model with the best val performance")
        #         self.load_model(self.output_dir)
        #     else:
        #         print("Deploy the last-epoch model")
        #     self.test()

        # Show elapsed time
        # elapsed = round(time.time() - self.time_start)
        # elapsed = str(datetime.timedelta(seconds=elapsed))
        # print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def test_fa(self, split=None):
        """A generic testing pipeline under faults."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference_fa(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)


        # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        # std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)
        # normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        #
        # surrogate = ClipModel(model=self.clip_model.visual, num_classes=2).to(self.device)
        # attacker = PGD(16 / 255., preprocess=normalize, num_iters=10)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_epoch_fa(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)


        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_fa(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain







#To get the roc, we need to first get the score
import argparse
import logging
import os 
import torch
import sys
sys.path.append('/content/drive/MyDrive/RankSVM_for_SLR')
from RankMamba.train_document import configure_model_and_tokenizer
from RankMamba.train_document import print_trainable_parameters
from RankMamba.train_document import load_from_trained
from RankMamba.train_document import configure_training_dataset

from transformers import AdamW
from RankMamba.train_document import train_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--t5_encoder", action="store_true", help="specify if using T5EncoderModel")

    parser.add_argument("--load_from_trained", default='True', help="declare if we load from existing checkpoint")
    parser.add_argument("--model_ckpt", type=str, help="use pytorch.bin if autoregressive model")

    parser.add_argument("--input_dir", type=str, default="/home/zhichao/msmarco_document")
    parser.add_argument("--triples", type=str, default="train_samples_lce.tsv")
    parser.add_argument("--lce_size", type=int, default=8)
    parser.add_argument('--experiment_root', type=str, default='./')

    # model specifics
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--pooling_method', type=str, default='eos-pooling', choices=['mean-pooling','cls-pooling','eos-pooling'])

    parser.add_argument('--flash_attention', action="store_true")
    parser.add_argument('--lora', action="store_true")
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)


    # training specifics
    parser.add_argument('--train_batch_size', type=int, default=16, help="total forward sequences is 8xbatch_size")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--training_steps', type=int, default=1e10)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--fold',type=int,default=0)

    # optimizer specifics
    parser.add_argument('--get_score',help='whether or not this function is used to get the score out of the final classification')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--disable_bias', action="store_true")
    parser.add_argument('--scheduler', type=str, default='warmuplinear')
    parser.add_argument('--warmup_steps', type=int, default=1e3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--cand_length', type=int, default=400)
    parser.add_argument('--query_length', type=int, default=110)
    parser.add_argument('--do_train', default=True,type=bool)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--eval_dataset', type=str, help="choose from dev, dl19, dl20, separate with comma")
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--ranklist', type=str, default='firstp.run')
    parser.add_argument('--feature_extraction',type=bool)
    parser.add_argument('--logger', type=str, default="default_logging.log")

    args = parser.parse_args()
    args.save_dest = os.path.join(args.experiment_root, "ckpt")
    #step1. Check if the model architecture is autoregressive
    if "opt" in args.model_name_or_path.lower() or "pythia" in args.model_name_or_path.lower() or "mamba" in args.model_name_or_path.lower() or "gpt2" in args.model_name_or_path:
        args.is_autoregressive = True
    else:
        args.is_autoregressive = False

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=args.logger, 
        filemode='a',
        )
    logger = logging.getLogger(__name__)
    logger.info("\n\n")
    for k, v in vars(args).items():
        logger.info(f"{k} -> {v}")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer, model = configure_model_and_tokenizer(model_name_or_path=args.model_name_or_path, args=args)
    print_trainable_parameters(model)

    if args.load_from_trained:
        assert args.model_ckpt is not None, "torch ckpt need to be specified if we load_from_trained"
        _, model = load_from_trained(args=args, initialized_model=model)
        print(f"loaded model ckpt from {args.model_ckpt}")

    # prepare document collection

    if args.do_train=='True':
        print('start training')
        trainset = configure_training_dataset(args=args, tokenizer=tokenizer)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            shuffle=False, 
            batch_size=args.train_batch_size, 
            collate_fn=trainset.collate_fn,
            #num_workers=2,
            num_workers=0,
            pin_memory=True
        )

        optimizer = optimizer = AdamW(model.parameters(), lr=1e-5,
                             weight_decay=0)

        # start training
        train_classification(
            tokenizer=tokenizer, 
            model=model, 
            train_loader=train_loader, 
            device=DEVICE, 
            optimizer=optimizer, 
            args=args,
            logger=logger
            )
    if args.get_score==True:
        print('start feature extraction')
        trainset = configure_training_dataset(args=args, tokenizer=tokenizer)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            shuffle=False, 
            batch_size=args.train_batch_size, 
            collate_fn=trainset.collate_fn,
            #num_workers=2,
            num_workers=0,
            pin_memory=True
        )
        optimizer = optimizer = AdamW(model.parameters(), lr=1e-5,
                             weight_decay=0)

        # start training
        train_classification(
            tokenizer=tokenizer, 
            model=model, 
            train_loader=train_loader, 
            device=DEVICE, 
            optimizer=optimizer, 
            args=args,
            logger=logger
            )
    
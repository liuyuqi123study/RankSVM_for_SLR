import os
import sys
import time
import csv
import argparse
import logging
import json
from torch.optim import lr_scheduler
from transformers import AdamW
import json 

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
import sys
sys.path.append('/content/drive/MyDrive/RankSVM_for_SLR/RankMamba')
from ranking_dataset import LCEDatasetCausalLM, LCEDatasetMaskedLM, LCEDatasetSeq2SeqLM
from utils import read_ranklist, read_qrels, configure_eval_dataset
from utils import read_validset
from utils import get_eval_batch
from utils import flatten_concatenation
from utils import load_lce_triples
from utils import save_model
from utils import load_from_trained

def nested2device(model, device):
    model.base_model = model.base_model.to(device)
    model.regressor = model.regressor.to(device)

    return model

def print_trainable_parameters(model):
    all_params, trainable_params = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}")


def configure_optimizer(model, disable_bias=False, lr=2e-5):
    if disable_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr)


def configure_model(model_name_or_path, tokenizer, args):
    if "opt" in model_name_or_path:
        from model import configure_opt_model
        model = configure_opt_model(model_name_or_path, tokenizer, args)
    elif "pythia" in model_name_or_path:
        from model import configure_pythia_model
        model = configure_pythia_model(model_name_or_path, tokenizer, args)
    elif "gpt2" in model_name_or_path:
        from model import configure_gpt2_model
        model = configure_gpt2_model(model_name_or_path, tokenizer, args)
    elif "mamba" in model_name_or_path:
        from model import configure_mamba_model
        model = configure_mamba_model(model_name_or_path, tokenizer, args)
    elif "t5" in model_name_or_path:
        from model import configure_t5_model
        model = configure_t5_model(model_name_or_path, tokenizer, args)
    elif "deberta" in model_name_or_path:
        from model import configure_deberta_model
        model = configure_deberta_model(model_name_or_path, tokenizer, args)
    elif "bert" in model_name_or_path:
        from model import configure_bert_model
        model = configure_bert_model(model_name_or_path, tokenizer, args)
    else:
        raise Exception("unexpected model name")

    return model

def configure_tokenizer(args,model_name_or_path):
    p_prefix, q_prefix = configure_special_tokens(model_name_or_path)
    if "opt" in model_name_or_path.lower() or "mamba" in model_name_or_path.lower() or "pythia" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id, tokenizer.pad_token = tokenizer.eos_token_id, tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        new_tokens = [p_prefix, q_prefix]
        tokenizer.add_tokens(list(new_tokens))
        
    return tokenizer

def configure_special_tokens(model_name_or_path):
    if "opt" in model_name_or_path.lower() or "mamba" in model_name_or_path.lower() or "pythia" in model_name_or_path.lower():
        p_prefix = "Document: "
        q_prefix = "Query: "
    else:
        p_prefix = "[passage]"
        q_prefix = "[query]"
    return p_prefix, q_prefix

def configure_model_and_tokenizer(model_name_or_path, args=None):
    tokenizer = configure_tokenizer(args,model_name_or_path)
    model = configure_model(model_name_or_path=model_name_or_path, tokenizer=tokenizer, args=args)
    if not args.is_autoregressive:
        model.base_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def configure_training_dataset(args, tokenizer,collection=None, queries=None, lce_dataset=None,mode='train'):
    if "t5" in args.model_name_or_path:
        return LCEDatasetSeq2SeqLM(collection=collection, queries=queries, dataset=lce_dataset, tokenizer=tokenizer, max_length=args.max_length)
    elif args.is_autoregressive:
        return LCEDatasetCausalLM(args=args,collection=collection,tokenizer=tokenizer, max_length=args.max_length,mode=mode)
    else:
        return LCEDatasetMaskedLM(collection=collection, queries=queries, dataset=lce_dataset, tokenizer=tokenizer, max_length=args.max_length)


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    if scheduler == 'warmuplinear':
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    if scheduler == 'constant':
        from transformers import get_constant_schedule
        return get_constant_schedule(optimizer)
    if scheduler == 'constantlinear':
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

def get_prediction(tokenizer, model, batch_input, args, device, extraction_feature=False):
    #assert isinstance(batch_input, list), 'wrong input type, force exit!'
    #tokenized_input = format_test_batch(batch_input, tokenizer, args.is_autoregressive)
    model = nested2device(model, device)
    model = model.half()

    with torch.no_grad():
        logits = model.forward(input_ids=batch_input.input_ids.to(device), attention_mask=batch_input.attention_mask.to(device),extraction_feature=extraction_feature)
    
    return logits
def format_test_batch(batch, tokenizer, is_autoregressive=True):
    input_pretokenized = []
    if "t5" in tokenizer.name_or_path:
        sep_token = tokenizer.sep_token
        bos_token = tokenizer.pad_token
        for i, row in enumerate(batch):
            input_pretokenized.append(bos_token+row[0]+row[1])
        return tokenizer(input_pretokenized, padding=True, truncation=True, max_length=512, return_tensors="pt")

    elif is_autoregressive:
        for i, row in enumerate(batch):
            document = tokenizer(row[1], truncation=True, max_length=768-50)  # currently hardcoded
            truncated_document = tokenizer.decode(document.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            input_pretokenized.append(truncated_document+"\n\n"+row[0]+tokenizer.eos_token)
        return tokenizer(input_pretokenized, padding=True, truncation=True, max_length=768, return_tensors="pt")
    else:
        sep_token = tokenizer.sep_token
        for i, row in enumerate(batch):
            input_pretokenized.append(row[0]+sep_token+row[1])
        return tokenizer(input_pretokenized, padding=True, truncation=True, max_length=512, return_tensors="pt")
        

def train_classification(
    tokenizer, 
    model, 
    train_loader, 
    device, 
    optimizer, 
    args,
    logger=None
    ):
    if args.do_train=='True':
        total_training_steps = min(args.training_steps, len(train_loader)*args.epochs)
        print(f"total training steps -> {total_training_steps}")
        save_milestone = total_training_steps // 10
        model_save_name = args.model_name_or_path.replace("/", "-")
        warmup_steps = max(args.warmup_steps, int(total_training_steps*args.warmup_ratio))
        scheduler = get_scheduler(optimizer, args.scheduler, args.warmup_steps, total_training_steps)


        loss_fct = torch.nn.CrossEntropyLoss()
        writer = SummaryWriter()

        # check_model_parameters(model)
        model = nested2device(model, device)
        model.train()
        
        train_steps = 0
        accumulated_loss = 0.
        flag = True
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        while flag:
            for epoch_id in range(args.epochs):
                model.train()
                current_epoch=epoch_id
                exp_lr_scheduler.step(current_epoch)
                if "t5" in model.config._name_or_path or "opt" in model.config._name_or_path or "pythia" in model.config._name_or_path:
                    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
                    autocast_dtype = torch.bfloat16
                else:
                    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
                    autocast_dtype = torch.float16
                for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"training epoch {epoch_id+1}", disable=args.disable_tqdm):
            
                    if train_steps > total_training_steps:
                        flag = False
                        break
                    with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=args.fp16):
                        output = model.forward(
                            input_ids=batch['inputx'].input_ids.to(device),
                            attention_mask=batch['inputx'].attention_mask.to(device),
                        )
                        
                        logits=output # by default this is set to 8, but can be changed to 16 as well
                        labels = batch['labels'].to(logits.device)  # (bz)
                        loss = loss_fct(logits, labels)
                    accumulated_loss+=loss
                    loss.backward()
        
                    train_steps+=1
                    optimizer.step()
                    optimizer.zero_grad()  # zero out the accumulated optimizer grad
                    if (train_steps) % 100 == 0:
                        print(f"\naverage loss -> {accumulated_loss/(train_steps):.2f}")

                    
                if args.do_eval:
        
                    evalset= configure_training_dataset(args=args, tokenizer=tokenizer,mode='eval')

                    model.to(DEVICE)
                    with open("prediction_"+str(args.fold)+'batch_size'+str(args.train_batch_size)+str(epoch_id)+'cand_length'+str(args.cand_length)+'learning_rate_1.json', "w") as fout:
                        tsv_writer = csv.writer(fout, delimiter=" ")
                    
                        model.eval()
                    
                        eval_loader=torch.utils.data.DataLoader(
                    evalset, 
                    shuffle=False, 
                    batch_size=args.eval_batch_size, 
                    collate_fn=evalset.collate_fn,
                    #num_workers=2,
                    num_workers=0,
                    pin_memory=True
                )
                        res_scores=[]
                        for batch_idx, batch in tqdm(enumerate(eval_loader), disable=args.disable_tqdm):
                            batch_scores = get_prediction(tokenizer, model, batch['inputx'], args, DEVICE)
                            score=torch.softmax(batch_scores,dim=1)
                            res_scores+=list(zip(batch['index'],score[:,1].tolist()))
                
                        predictions={}
                        for res in res_scores:
                            
            
                                if res[0][0] not in predictions:
                                    predictions[res[0][0]] = []
                                predictions[res[0][0]].append((res[0][1], res[1]))

                        for key in predictions:
                                predictions[key].sort(key = lambda x:x[1], reverse = True)
                                predictions[key] = [int(res[0]) for res in predictions[key]]
                        print(json.dumps(predictions),file=fout)
                        
                    fout.close()
            
                #Save the model
                save_dest = os.path.join(args.save_dest, str(args.fold),f"cand_length_{args.cand_length}_learning_rate_{args.lr}_{model_save_name}_batch_size_{args.train_batch_size}")
                print(f"saving to {save_dest}")
                save_model(model=model, save_dest=save_dest)
                tokenizer.save_pretrained(save_dest)
            flag=False
    else:
        print('feature extraction running')
        evalset= configure_training_dataset(args=args, tokenizer=tokenizer,mode='eval')
        model.to(device)
        model.eval()
        eval_loader=torch.utils.data.DataLoader(
                    evalset, 
                    shuffle=False, 
                    batch_size=args.eval_batch_size, 
                    collate_fn=evalset.collate_fn,
                    #num_workers=2,
                    num_workers=0,
                    pin_memory=True
                )
        for batch_idx, batch in tqdm(enumerate(train_loader), disable=args.disable_tqdm):
                            pooled_output_trains = get_prediction(tokenizer, model, batch['inputx'], args, device,extraction_feature=True).cpu().numpy.tolist()
                            labels=batch['labels']
                            indexes=batch['index']
                            features=[' '.join([str(ind)+':'+str(value) for (ind,value) in enumerate(pooled_output_train)]) for pooled_output_train in pooled_output_trains]
        
                           
                            contents=[str(label)+' '+'qid:'+str(index[0]+5181)+' '+feature+'#'+str(index[1]) for(label, index,feature) in zip(labels,indexes,features)]
                            with open('train'+str(args.fold)+'.dat', 'a') as file:
                                json.dump(contents, file)
        for batch_idx, batch in tqdm(enumerate(eval_loader), disable=args.disable_tqdm):
                            pooled_output_tests = get_prediction(tokenizer, model, batch['inputx'], args, device,extraction_feature=True).cpu().numpy.tolist()
                            labels=batch['labels']
                            indexes=batch['index']
                            features=[' '.join([str(ind)+':'+str(value) for (ind,value) in enumerate(pooled_output_test)]) for pooled_output_test in pooled_output_tests]
        
                           
                            contents+=[[str(label)+' '+'qid:'+str(index[0]+5181)+' '+feature+'#'+str(index[1])] for(label, index,feature) in zip(labels,indexes,features)]
                            with open('train'+str(args.fold)+'.dat', 'a') as file:
                                json.dump(contents, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--t5_encoder", action="store_true", help="specify if using T5EncoderModel")

    parser.add_argument("--load_from_trained", action="store_true", help="declare if we load from existing checkpoint")
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
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--disable_bias', action="store_true")
    parser.add_argument('--scheduler', type=str, default='warmuplinear')
    parser.add_argument('--warmup_steps', type=int, default=1e3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--cand_length', type=int, default=400)
    parser.add_argument('--query_length', type=int, default=110)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--eval_dataset', type=str, help="choose from dev, dl19, dl20, separate with comma")
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--ranklist', type=str, default='firstp.run')

    parser.add_argument('--logger', type=str, default="default_logging.log")

    args = parser.parse_args()
    args.save_dest = os.path.join(args.experiment_root, "ckpt")
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
    
    if args.do_train:
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
    
    
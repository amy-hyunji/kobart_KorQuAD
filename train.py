import logging
import argparse
import torch
import os
import transformers 
import pytorch_lightning as pl
import yaml

from pytorch_lightning import loggers as pl_loggers 
from torch.utils.data import DataLoader
from transformers import BartForQuestionAnswering, PreTrainedTokenizerFast

from dataloader import QADataModule
from model import KoBART_QA

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<dense-vectors>", "<tokens>", "<verb>", "<ARG0>", "<ARG1>", "<global-dense-vectors>"],
    "pad_token": "<pad>",
    "bos_token": "<bos>", 
    "eos_token": "<eos>", 
    "cls_token": "<cls>",
    "sep_token": "<sep>",
    "unk_token": "<unk>",
}

parser = argparse.ArgumentParser(description="KoBART KoSQuAD")

parser.add_argument("--max_len", type=int, default=384) #384
parser.add_argument("--max_epochs", type=int, default=35)
parser.add_argument("--train_path", type=str, default="./data/KorQuAD_v1.0_train.json")
parser.add_argument("--test_path", type=str, default="./data/KorQuAD_v1.0_dev.json")
parser.add_argument("--checkpoint_path", type=str, default="./output/last_ckpt.ckpt")
parser.add_argument("--hparams", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32) #32
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--infer_one", action='store_true')
parser.add_argument("--default_root_dir", type=str, default="./")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)

args = parser.parse_args()
logging.info(args)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart", cls_token="<s>", sep_token="</s>")
print(f"cls id: {tokenizer.cls_token_id}")
print(f"sep id: {tokenizer.sep_token_id}")

# output keys of model -> start_logits, end_logits, past_key_values, encoder_last_hidden_state

"""
if args.infer_one:
    f = open(args.hparams)
    hparams = yaml.load(f)
    hparams['checkpoint_path'] = args.checkpoint_path
    hparams['infer_one'] = args.infer_one
    print(f"hparams: {hparams}")
    model = KoBART_QA(hparams)
else:
    model = KoBART_QA(args)
"""
print(f"args: {args}")
model = KoBART_QA(args)

if args.infer_one:
    model.model.eval()
    while(1):
        c = input('context> ').strip()
        if (c == 'quit'): break
        print(" ")
        q = input('question> ').strip()
        if (q == 'quit'): break
        ret = model.infer_one(tokenizer, c, q)

else:
    dm = QADataModule(args, tokenizer)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', 
                                dirpath=args.default_root_dir, 
                                filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                verbose=True,
                                save_last=True,
                                mode="min",
                                save_top_k=4,
                                prefix='kobart_qa')
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                callbacks=[checkpoint_callback, lr_logger])  
    trainer.fit(model, dm)


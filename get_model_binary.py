import argparse
from model import KoBART_QA  
from transformers.models.bart import BartForQuestionAnswering 
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str, help="path to hparams.yaml saved by training")
parser.add_argument("--model_binary", default=None, type=str, help="path to .ckpt saved by training")
parser.add_argument("--output_dir", default='kobart_qa', type=str, help="output file to save")
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)
    
inf = KoBART_QA.load_from_checkpoint(args.model_binary, hparams=hparams)

inf.model.save_pretrained(args.output_dir)


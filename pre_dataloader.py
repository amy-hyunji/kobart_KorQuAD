"""
Codes for pre-processing dataset for Squad v2.0
"""

import json
import sys
import os
import argparse
import pandas as pd

from pathlib import Path
from transformers import PreTrainedTokenizerFast

from utils import add_end_idx, text_processing

def read_squad(_file):
    contexts = []
    questions = []
    answers = []

    with open(_file, 'rb') as f:
        try:
            squad_dict = json.load(f)
        except:
            print(f"************* Error on file: {_file} *****************") 
            return None

    for group in squad_dict['data']:
        context = group['context']
        for qa in group['qas']:
            question = qa['question']
            answer = qa['answer']

            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    add_end_idx(answers, contexts)
    assert (len(contexts)==len(questions)==len(answers))
    print(f"# of total pairs in file {_file.split('/')[-1]}: {len(contexts)}")
    return {'context': contexts, 'question': questions, 'answer': answers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KoBART KorSquAD 2.0")
    parser.add_argument("--data_path", type=str, default="./data/ver_2.0/train")
    parser.add_argument("--save_path", type=str, default="./data/cached_train_768")
    parser.add_argument("--max_len", type=int, default=768)
    parser.add_argument("--cleanse", action='store_true')
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart", cls_token="<s>", sep_token="</s>")
    if not os.path.exists(args.save_path): 
        os.mkdir(args.save_path)
    
    file_list = os.listdir(args.data_path)
    saved_list = os.listdir(args.save_path)
    for _file in file_list:
        for s_file in saved_list:
            if (_file.split(".json")[0] in s_file):
                print(f"## {_file} already exists!! == {s_file}")
                continue
        data_dict = read_squad(os.path.join(args.data_path, _file))
        if (data_dict is None): continue
        data = text_processing(data_dict, args, tokenizer)

        # save tokenized result
        df = pd.DataFrame(data)
        num = len(data['input_ids'])
        file_name = str(num) + "_" + _file.split(".json")[0]+".pkl"
        df.to_pickle(os.path.join(args.save_path, file_name))
        print(f"Done Saving {os.path.join(args.save_path, file_name)}")

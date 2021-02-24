import json
import sys
import os
import torch
import pickle
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader, Dataset 

from utils import add_end_idx, text_processing

debug = False 
k_space = "▁", 

class QADataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.max_len = args.max_len
    def setup(self, stage):
        if self.args.shuffle: 
            print("@@@@@ shuffling train dataset")
        else:
            print("@@@@@ NOT shuffling train dataset")
        if self.args.squad_ver == 1:
            self.qa_train = QA_dataset1(self.args, self.tokenizer, "train")
            self.qa_val = QA_dataset1(self.args, self.tokenizer, "test")
        elif self.args.squad_ver == 2:
            self.qa_train = QA_dataset2(self.args, "train")
            self.qa_val = QA_dataset2(self.args, "test")
        else:
            print(f"ERROR: inappropriate squad version - {self.args.squad_ver}")
            sys.exit(-1)
    def train_dataloader(self):
        qa_train = DataLoader(self.qa_train, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=self.args.shuffle)
        return qa_train
    def val_dataloader(self):
        qa_val = DataLoader(self.qa_val, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=False)
        return qa_val

# for KorQuAD 1.0
class QA_dataset1(Dataset):
    def __init__(self, args, tokenizer, state="train"):
        if state == "train":
            print(f"train_path: {args.train_path}")
            data_dict = self.read_squad(args.train_path)
        elif state == "test":
            print(f"test_path: {args.test_path}")
            data_dict = self.read_squad(args.test_path)
        else:
            print("check the state argument in QA_dataset")
            sys.exit(-1)
        data = text_processing(data_dict, args, tokenizer)
        self.tokenizer = tokenizer
        self.input = data['input_ids']
        self.attn_mask = data['attn_mask_ids'] 
        self.label_start = data['label_start'] 
        self.label_end = data['label_end']
        print(f"***** state: {state}, # of dataset: {len(self.input)}")
        assert (len(self.input) == len(self.attn_mask) == len(self.label_start) == len(self.label_end))

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        x = self.input[idx]
        z = self.attn_mask[idx] 
        label_s = self.label_start[idx].view(-1)
        label_e = self.label_end[idx].view(-1)
        return {'input_ids': x, 'attention_mask': z, 'label_s': label_s, 'label_e': label_e}

    def read_squad(self, path):
        contexts = []
        questions = []
        answers = []

        file_list = os.listdir(path)
        for _file in file_list:
            print(f"Working on ... {_file}")
            with open(os.path.join(path, _file), 'rb') as f:
                try:
                    squad_dict = json.load(f)
                except:
                    print(f"Error on file: {_file}")
                    continue

            for group in squad_dict['data']:
                if 'paragraphs' in group.keys():
                    for passage in group['paragraphs']:
                        context = passage['context']
                        for qa in passage['qas']:
                            question = qa['question']
                            # for no answer case
                            if 'answers' not in qa.keys():
                                answer = {'answer_start': 0, 'answer_end': 0, 'text': None}
                                contexts.append(context)
                                questions.append(question)
                                answers.append(answer)
                            else:
                                for answer in qa['answers']:
                                    if answer['text'][0] == ' ':
                                        continue
                                    contexts.append(context)
                                    questions.append(question)
                                    answers.append(answer)
        add_end_idx(answers, contexts)
        assert (len(contexts)==len(questions)==len(answers))
        print(f"# of total contexts: {len(contexts)}") #60407 for ver1.0
        return {'context': contexts, 'question': questions, 'answer': answers}
      
# for KorQuAD 2.0
class QA_dataset2(Dataset):
    def __init__(self, args, state="train"):
        print("### Working on KorQuAD 2.0")
        self.path = None
        self.df = None
        if state == "train":
            print(f"train_path: {args.train_path}") 
            self.data_list = self.get_path_list(args.train_path)
        elif state == "test":
            print(f"test_path: {args.test_path}")
            self.data_list = self.get_path_list(args.test_path)
        else:
            print("check the state argument in QA_dataset")
            sys.exit(-1)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, num = self.data_list[idx]
        if path != self.path:
            self.path = path
            self.df = pd.read_pickle(path)
        input_ids = list(self.df['input_ids'])[num]
        attn_mask_ids = list(self.df['attn_mask_ids'])[num]
        label_start = list(self.df['label_start'])[num]
        label_end = list(self.df['label_end'])[num]
        return {'input_ids': input_ids, 'attention_mask': attn_mask_ids, 'label_s': label_start, 'label_e': label_end}

    def get_path_list(self, path):
        # create ret_list which is a list of [path_to_file, row_num]
        file_list = os.listdir(path)
        ret_list = []
        for _file in file_list:
            num = int(_file.split("_")[0])
            for i in range(num):
                ret_list.append([os.path.join(path, _file), i])
        return ret_list
    

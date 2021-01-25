import json
import sys
import torch
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader, Dataset 

"""
[input]
- input_ids (tokenized data)
- attention_mask_ids (1 for sentence, 0 for padded parts)
- format --> [CLS] + [context] + [SEP]*2 + [question] + [SEP] + [PAD]

* when context is larger than max_len --> cut to chunk 
* label_start and label_end will be [CLS] if does not exist
"""

class QADataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.max_len = args.max_len
    def setup(self, stage):
        self.qa_train = QA_dataset(self.args, self.tokenizer, "train")
        self.qa_val = QA_dataset(self.args, self.tokenizer, "test")
    def train_dataloader(self):
        qa_train = DataLoader(self.qa_train, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=True)
        return qa_train
    def val_dataloader(self):
        qa_val = DataLoader(self.qa_train, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=False)
        return qa_val

class QA_dataset(Dataset):
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
        data = self.text_processing(data_dict, args, tokenizer)
        self.tokenizer = tokenizer
        self.input = data['input_ids']
        self.attn_mask = data['attn_mask_ids'] 
        self.label_start = data['label_start'] 
        self.label_end = data['label_end']
  
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        x = self.input[idx]
        z = self.attn_mask[idx] 
        label_s = self.label_start[idx].view(-1)
        label_e = self.label_end[idx].view(-1)
        #return x, y, z, label_s, label_e
        return {'input_ids': x, 'attention_mask': z, 'label_s': label_s, 'label_e': label_e}


    #### remove torch_type_ids -> add decoder_input_ids, decoder_attention_mask

    def text_processing(self, data, args, tokenizer):
        input_ids = []
        attn_mask_ids = []
        label_start = [] # char start idx in context 
        label_end = []   # char end idx in context
        max_len = args.max_len

        for i in range(len(data)):

            context = data['context'][i]
            question = data['question'][i] 
            answer = data['answer'][i]
            start_idx = answer['answer_start']
            end_idx = answer['answer_end']
            answer_text = answer['text']

            c = tokenizer.encode(context, add_special_tokens=False)
            q = tokenizer.encode(question, add_special_tokens=False)
            a = tokenizer.encode(answer_text, add_special_tokens=False)

            """
            if (i==0):
                print(f"context: {context}")
                print(f"question: {question}")
                print(f"answer: {answer}")
                print(f"start_idx: {start_idx}")
                print(f"end_idx: {end_idx}")
                print(f"answer_text: {answer_text}")
                print(f"c: {c}") 
                print(f"q: {q}") 
                print(f"a: {a}") 
            """
            spair_len = max_len - 4 - len(q) # length left for context 
            assert (spair_len > 0)

            chunk = []
            if (len(c) <= spair_len):
                # no problem. just make the format 
                chunk.append(c)

            else:
                # need to divide context into chunk
                iter_num = len(c)//spair_len + 1 if len(c)%spair_len != 0 else len(c)//spair_len
                for i in range(iter_num):
                    chunk.append(c[i*spair_len:(i+1)*spair_len])

            stack_idx = 0
            for i, _chunk in enumerate(chunk):
                # end = start = 0 if not exist in _chunk
                # take care with the fact that idx starts from 0
              
                _chunk_len = len(_chunk) + len(q) + 4 
                _input_ids = [tokenizer.cls_token_id] + q + [tokenizer.sep_token_id]*2 + _chunk + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_len-_chunk_len)  
                assert(len(_input_ids) <= max_len)
                _attn_mask_ids = [1]*_chunk_len + [0]*(max_len-_chunk_len)

                _start_idx = start_idx - stack_idx      # idx in current chunk
                _end_idx = end_idx - stack_idx          # idx in current chunk
                
                # already extracted answer from previous chunk
                if (_start_idx < 0 and _end_idx < 0):
                    _label_start = 0
                    _label_end = 0
           
                # second chunk with partial answer
                elif (_start_idx < 0 and _end_idx >= 0):
                    _label_start = 0
                    _label_end = _end_idx

                elif (_start_idx >= 0 and _end_idx < 0):
                    print("Shouldn't be here!\ncontext: {context}\nchunk: {_chunk}\nanswer: {answer}")
                    sys.exit(-1)

                else:
                    # answer in later chunk 
                    if len(_chunk) <= _start_idx:
                        _label_start = 0
                        _label_end = 0
                    elif _start_idx < len(_chunk):
                        # first chunk with partial answer
                        if len(_chunk) <= _end_idx:
                            _label_start = _start_idx
                            _label_end = len(_chunk)-1
                        # all in current chunk
                        else:
                            _label_start = _start_idx
                            _label_end = _end_idx
                stack_idx += len(_chunk)
                
                input_ids.append(torch.tensor(_input_ids, dtype=torch.long) )
                attn_mask_ids.append(torch.tensor(_attn_mask_ids, dtype=torch.long))
                assert (_label_start >= 0 and _label_end >= 0 and _label_end <= len(_chunk)-1)
                label_start.append(torch.tensor(_label_start, dtype=torch.long))
                label_end.append(torch.tensor(_label_end, dtype=torch.long))

        assert(len(input_ids) == len(attn_mask_ids) == len(label_start) == len(label_end))

        return {'input_ids': input_ids, 'attn_mask_ids': attn_mask_ids, 'label_start': label_start, 'label_end': label_end}    

    def add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two - fix this 
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx 
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx-1
                answer['answer_end'] = end_idx-1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx-2
                answer['answer_end'] = end_idx-2 
        assert (answer['answer_start'] < len(contexts) and answer['answer_end'] < len(contexts))
        return

    def read_squad(self, path):
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        self.add_end_idx(answers, contexts)

        assert (len(contexts)==len(questions)==len(answers))
        print(f"# of contexts: {len(contexts)}")
        return {'context': contexts, 'question': questions, 'answer': answers}
        

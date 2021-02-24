import torch
import argparse

from transformers import BartForQuestionAnswering, PreTrainedTokenizerFast 
from konlpy.tag import Mecab

class KoBART_QA():
    def __init__(self, ckpt_path="./kobart_qa", max_len=1024):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BartForQuestionAnswering.from_pretrained(ckpt_path).to(self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart", cls_token="<s>", sep_token="</s>")
        self.model.eval()
        self.max_len = max_len
        self.mecab = Mecab()

    def get_chunk(self, context, question):
        c = self.tokenizer.encode(context, add_special_tokens=False)
        q = self.tokenizer.encode(question, add_special_tokens=False)
        
        spair_len = self.max_len-4-len(q)
        assert(spair_len > 0)

        chunk = []
        question = []
        if (len(c) <= spair_len):
            # no need to divide into chunk. just make the format
            chunk.append(c)
            question.append(q)
            chunk_num = 1
        else:
            # length is over max_len. need to divide context into chunk
            iter_num = len(c)//spair_len+1 if len(c)%spair_len!=0 else len(c)//spair_len
            for i in range(iter_num):
                chunk.append(c[i*spair_len:(i+1)*spair_len])
                question.append(q)
            chunk_num = iter_num

        return chunk, question, chunk_num

    def data_process(self, chunk, q_list):
        assert(len(chunk) == len(q_list))
        ret_dict = {'input_ids': [], 'attention_mask': []}
        for i, (_chunk, _q) in enumerate(zip(chunk, q_list)):
            total_len = 4 + len(_chunk) + len(_q)
            attention_mask = [1]*total_len + [0]*(self.max_len-total_len)
            input_ids = [self.tokenizer.cls_token_id] + _chunk + [self.tokenizer.sep_token_id]*2 + _q + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id]*(self.max_len-total_len)
            ret_dict['input_ids'].append(input_ids)
            ret_dict['attention_mask'].append(attention_mask)
        ret_dict['input_ids'] = torch.tensor(ret_dict['input_ids']).to(self.device)
        ret_dict['attention_mask'] = torch.tensor(ret_dict['attention_mask']).to(self.device)
        return ret_dict

    def batch_infer(self, batch, skip_long):
        ret_list = []
        chunk = []
        question = []
        chunk_num = []
        for (c, q) in batch:
            _chunk, _q_list, _chunk_num = self.get_chunk(c, q)
            chunk += _chunk
            question += _q_list
            chunk_num.append(_chunk_num)
        data_dict = self.data_process(chunk, question)

        output = self.model(input_ids=data_dict['input_ids'], attention_mask=data_dict['attention_mask'])
        start_pts = torch.argmax(output.start_logits, dim=1)
        end_pts = torch.argmax(output.end_logits, dim=1)
        iter_num = 0
        for num in chunk_num:
            # one answer for one question 
            ret = ""
            t_list = []
            for _ in range(int(num)):
                t_ret = ""
                input_ids = list(data_dict['input_ids'][iter_num])
                start_pt = list(start_pts)[iter_num]
                end_pt = list(end_pts)[iter_num]
                iter_num += 1
                if (start_pt < end_pt):
                    ans = input_ids[start_pt:end_pt]
                    tok_list = self.tokenizer.convert_ids_to_tokens(ans)
                    for elem in tok_list:
                        t_ret += elem
                    t_ret = t_ret.replace("▁", " ")
                    t_ret = t_ret.replace("<s>", "")
                    t_ret = t_ret.replace("</s>", "")
                    if t_ret in t_list:
                        continue
                    else:
                        t_list.append(t_ret)
            for _t_ret in t_list:
                ret += _t_ret
            try:
                elem_list = self.mecab.pos(ret)
                print(elem_list)
                if elem_list[-1][1] in ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC', 'JX']:
                    ret = ret.replace(elem_list[-1][0], '')
            except:
                print("Pass Pos TAG")
            if skip_long and len(ret) > 150:
                ret_list.append("")
            else:
                ret_list.append(ret)
        return ret_list

    def infer(self, context, question):
        ret = ""
        chunk, q_list, chunk_num = self.get_chunk(context, question)
        data_dict = self.data_process(chunk, q_list)
        output = self.model(input_ids=data_dict['input_ids'], attention_mask=data_dict['attention_mask'])
        start_pts = torch.argmax(output.start_logits, dim=1)
        end_pts = torch.argmax(output.end_logits, dim=1)
        for i, (start_pt, end_pt) in enumerate(zip(list(start_pts), list(end_pts))):
            input_ids = list(data_dict['input_ids'][i])
            if (start_pt < end_pt):
                ans = input_ids[start_pt:end_pt]
                ret_list = self.tokenizer.convert_ids_to_tokens(ans)
                for elem in ret_list:
                    ret += elem
        ret = ret.replace("▁", " ")
        try:
            elem_list = self.mecab.pos(ret)
            if elem_list[-1][1] in ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC', 'JX']:
                ret = ret.replace(elem_list[-1][0], '')
        except:
            print("Pass Pos TAG")
        return ret


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=384, help="max length of the context chunk")
    parser.add_argument("--skip_long", action='store_true', help="skip long answers (threshold 150)")
    parser.add_argument("--batch", action='store_true', help="operate in batch")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--ckpt_path", type=str, default="./kobart_qa", help="path to binary file")
    args = parser.parse_args()
    
    QA_class = KoBART_QA(args.ckpt_path, args.max_len)
    
    if args.batch:
        batch = []
        c = input('context> ').strip()
        for i in range (args.batch_size):
            q = input('question> ').strip()
            batch.append([c, q])
        a_list = QA_class.batch_infer(batch, args.skip_long)
        for i, a in enumerate(a_list): 
            if a is None:
                print(f"[{i}]: No Answer!") 
            else:
                print(f"[{i}]: Answer {a}")
    else:
        while 1:
            c = input('context> ').strip()
            if c == "exit": exit()
            while 1:
                q = input('question> ').strip()
                if q == "exit":
                    break
                a = QA_class.infer(c, q)
                if a is None:
                    print("No Answer!") 
                else:
                    print(f"Answer: {a}")

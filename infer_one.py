import torch
from transformers import BartForQuestionAnswering, PreTrainedTokenizerFast 

class KoBART_QA():
    def __init__(self, ckpt_path="./kobart_qa", max_len=384):
        print(torch.cuda.is_available())
        self.model = BartForQuestionAnswering.from_pretrained(ckpt_path).cuda() 
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart", cls_token="<s>", sep_token="</s>")
        self.model.eval()
        self.device = 'cuda'
        self.max_len = max_len
    
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
        ret_dict['input_ids'] = torch.tensor(ret_dict['input_ids']).cuda()
        ret_dict['attention_mask'] = torch.tensor(ret_dict['attention_mask']).cuda()
        return ret_dict

    def batch_infer(self, batch):
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
            for _ in range(int(num)):
                input_ids = list(data_dict['input_ids'][iter_num])
                start_pt = list(start_pts)[iter_num]
                end_pt = list(end_pts)[iter_num]
                iter_num += 1
                if (start_pt < end_pt):
                    ans = input_ids[start_pt:end_pt]
                    tok_list = self.tokenizer.convert_ids_to_tokens(ans)
                    for elem in tok_list:
                        ret += elem
            ret = ret.replace("▁", " ")
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
        return ret


if __name__ == "__main__":
    QA_class = KoBART_QA()
    batch = True 
    
    if batch:
        batch = []
        c = input('context> ').strip()
        for i in range (3):
            q = input('question> ').strip()
            batch.append([c, q])
        a_list = QA_class.batch_infer(batch)
        for a in a_list: 
            if a is None:
                print("No Answer!") 
            else:
                print(f"Answer: {a}")
    else:
        c = input('context> ').strip()
        while 1:
            q = input('question> ').strip()
            a = QA_class.infer(c, q)
            if a is None:
                print("No Answer!") 
            else:
                print(f"Answer: {a}")
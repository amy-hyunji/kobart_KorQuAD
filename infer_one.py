import torch
from transformers import BartForQuestionAnswering, PreTrainedTokenizerFast 
from memory_profiler import profile

class KoBART_QA():
    def __init__(self, ckpt_path="./kobart_qa", max_len=384):
        print(torch.cuda.is_available())
        self.model = BartForQuestionAnswering.from_pretrained(ckpt_path).cuda() 
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart", cls_token="<s>", sep_token="</s>")
        self.model.eval()
        self.device = 'cuda'
        self.max_len = max_len
    def infer(self, context, question):
        c = self.tokenizer.encode(context, add_special_tokens=False)
        q = self.tokenizer.encode(question, add_special_tokens=False)
        
        spair_len = self.max_len-4-len(q)
        assert(spair_len > 0)

        chunk = []
        if (len(c) <= spair_len):
            # no need to divide into chunk. just make the format
            chunk.append(c)
        else:
            # length is over max_len. need to divide context into chunk
            iter_num = len(c)//spair_len+1 if len(c)%spair_len!=0 else len(c)//spair_len
            for i in range(iter_num):
                chunk.append(c[i*spair_len:(i+1)*spair_len])
        
        ret = ""
        for i, _chunk in enumerate(chunk):
            total_len = 4 + len(_chunk) + len(q)
            attention_mask = torch.tensor([[1]*total_len + [0]*(self.max_len-total_len)]).cuda()
            input_ids = torch.tensor([[self.tokenizer.cls_token_id] + _chunk + [self.tokenizer.sep_token_id]*2 + q + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id]*(self.max_len-total_len)]).cuda()
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_pt = torch.argmax(output.start_logits)
            end_pt = torch.argmax(output.end_logits)
            input_ids = input_ids[0].tolist()
            if (start_pt >= end_pt):
                return None 
            else:
                ans = input_ids[start_pt:end_pt]
                for elem in ans:
                    ret += self.tokenizer.convert_ids_to_tokens(elem)
        ret = ret.replace("▁", " ")
        return ret


if __name__ == "__main__":
    QA_class = KoBART_QA()
    c = input('context> ').strip()
    while (1):
        q = input('question> ').strip()
        a = QA_class.infer(c, q)
        if a is None:
            print("No Answer!") 
        else:
            print(f"Answer: {a}")

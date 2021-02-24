import pytorch_lightning as pl
import transformers
import torch

from transformers import BartForQuestionAnswering
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super().__init__()
        self.hparams = hparams
    
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        print(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers * self.hparams.accumulate_grad_batches) * self.hparams.max_epochs)
        print(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        print(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class KoBART_QA(Base):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        if not self.hparams.infer_one:
            print("### In train mode")
            if not self.hparams.checkpoint_path: 
                self.model = BartForQuestionAnswering.from_pretrained("hyunwoongko/kobart")
            else:
                print(f"### Loading ckpt from.. {self.hparams.checkpoint_path}")
                self.model = BartForQuestionAnswering.from_pretrained(self.hparams.checkpoint_path)
            self.model.train()
        else:
            print(f"### In eval mode! Loading from.. {self.hparams.checkpoint_path}")
            self.model = BartForQuestionAnswering.from_pretrained(self.hparams.checkpoint_path)
            self.model.eval()
        
        if torch.cuda.is_available():
            print("*** Working in GPU")
            self.gpu = True
            self.model.cuda()
        else:
            print("*** Working in CPU")
            self.gpu = False

    def forward(self, inputs):
        if self.gpu:
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            inputs['label_s'] = inputs['label_s'].cuda()
            inputs['label_e'] = inputs['label_e'].cuda()
        return self.model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            start_positions=inputs['label_s'],
                            end_positions=inputs['label_e'],
                            return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('val_step_loss', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

    """
    def infer_one(self, tokenizer, context, question):
        
        c = tokenizer.encode(context, add_special_tokens=False)
        q = tokenizer.encode(question, add_special_tokens=False)
        max_len = self.hparams.max_len
        print("### Max Length: {max_len}")
        
        spair_len = max_len-4-len(q)
        assert (spair_len > 0)

        chunk = []
        if (len(c) <= spair_len):
            # no problem. just make the format
            print(f"Done fitting into one chunk!")
            chunk.append(c)
        else:
            # need to divide context into chunk
            print(f"Dividing! Cannot fit into one chunk!")
            iter_num = len(c)//spair_len + 1 if len(c)%spair_len != 0 else len(c)//spair_len
            for i in range(iter_num):
                chunk.append(c[i*spair_len:(i+1)*spair_len])

        ret = ""
        for i, _chunk in enumerate(chunk):
            print("chunk: ", tokenizer.convert_ids_to_tokens(_chunk))
            total_len = 4 + len(_chunk) + len(q)
            input_ids = torch.tensor([[tokenizer.cls_token_id] + _chunk + [tokenizer.sep_token_id]*2 + q + [tokenizer.sep_token_id] + [tokenizer.pad_token_id]*(max_len-total_len)])
            attention_mask = torch.tensor([[1]*total_len + [0]*(max_len-total_len)])
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            assert (input_ids.shape == output.start_logits.shape == output.end_logits.shape)
            start_pt = torch.argmax(output.start_logits)
            end_pt = torch.argmax(output.end_logits)
            input_ids = input_ids[0].tolist()
            if (start_pt >= end_pt):
                if (start_pt==0 and end_pt==0):
                    print("No answer!!")
                else:
                    print("Error. End point should be larger than start point")
            else:
                ans = input_ids[start_pt:end_pt]
                print(tokenizer.convert_ids_to_tokens(ans))
                for elem in ans:
                    ret += tokenizer.convert_ids_to_tokens(elem)    
        print("#### final answer: ",ret.replace("▁", " "))
        return
    """

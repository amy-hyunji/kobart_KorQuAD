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
            self.model = BartForQuestionAnswering.from_pretrained("hyunwoongko/kobart")
            print("### In train mode")
            self.model.train()
        else:
            print(f"### In eval mode! Loading from.. {self.hparams.checkpoint_path}")
            self.model = BartForQuestionAnswering.from_pretrained(self.hparams.checkpoint_path)

    def forward(self, inputs):
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
        return loss

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

    def infer_one(self, tokenizer, context, question):
        c = tokenizer.encode(context, add_special_tokens=False)
        q = tokenizer.encode(question, add_special_tokens=False)
        max_len = self.hparams.max_len
        total_len = len(q)+len(c)+4
        input_ids = torch.tensor([[tokenizer.cls_token_id] + q + [tokenizer.sep_token_id]*2 + c + [tokenizer.sep_token_id]])
        attention_mask = torch.tensor([[1]*total_len])
        print("input_ids: ",input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        assert (input_ids.shape == output.start_logits.shape == output.end_logits.shape)
        start_pt = torch.argmax(output.start_logits)
        end_pt = torch.argmax(output.end_logits)
        input_ids = input_ids[0].tolist()
        print(f"start_pt: {start_pt}, end_pt: {end_pt}")
        if (start_pt > end_pt):
            print("Error. Start point is larger than end point")
        else:
            if (start_pt==0 and end_pt==0):
                print("No answer!!")
            else:
                ans = context[start_pt:end_pt]
                ret = ""
                for elem in ans:
                    ret += elem    
                print(ret.replace("▁", " "))
        return

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
        self.model = BartForQuestionAnswering.from_pretrained("hyunwoongko/kobart")
        self.model.train()

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type'],
                            start_position=inputs['label_s'],
                            end_position=inputs['label_e'],
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
        input_ids = [tokenizer.cls_token] + c + [tokenizer.sep_token_id] + q + [tokenizer.sep_token_id]
        res_ids = self.model(torch.tensor([input_ids]), 
                    max_length=self.hparams.max_len, 
                    num_beams=5,
                    eos_token_id=tokenizer.sep_token_id,
                    bad_token_id=[[tokenizer.unk_token_id]])
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace(tokenizer.cls_token, '').replace(tokenizer.sep_token, '')

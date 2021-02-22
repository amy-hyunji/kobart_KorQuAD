# kobart_KorQuAD

Q&A model using kobart released from SKT-AI
* [korquad](https://korquad.github.io/category/1.0_KOR.html)
* [kobart](https://github.com/SKT-AI/KoBART)

## train & test
```
bash run_train.sh
```

## Evaluation
change **hparams and model_binary** arguments in `run_binary` file and run
```
bash run_binary.sh
```
-> this will create `config.json` and `pytorch_model.bin` under `kobart_qa` file 

### Real Time Inference
```
python infer_one.py
```
`context>`: insert context  
`question>`: insert question

## Evaluation
```
python predict.py
```

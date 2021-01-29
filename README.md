# kobart_KorQuAD

## train & test
```
bash run_train.sh
```

## Real Time Inference
change **hparams and model_binary** arguments in `run_binary` file and run
```
bash run_binary.sh
```
-> this will create `config.json` and `pytorch_model.bin` under `kobart_qa` file 

  
```
bash run_eval.sh
```
`context>`: insert context  
`question>`: insert question

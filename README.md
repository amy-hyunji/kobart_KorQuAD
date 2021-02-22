# kobart_KorQuAD

Q&A model using kobart released from SKT-AI
* [korquad v1](https://korquad.github.io/category/1.0_KOR.html)
* [korquad v2](https://korquad.github.io/)
* [kobart](https://github.com/SKT-AI/KoBART)

* 추가적으로 학습시 사용한 데이터셋
  * [AI-HUB 기계독해](https://aihub.or.kr/aidata/86)
  * [AI-HUB 일반상식](https://aihub.or.kr/aidata/84)

## Setting
```
* transformers=4.0.0
* mecab
* konlpy
```

## Train
`bash script/run_train.sh`
* parameters
```
  --max_len MAX_LEN     max length of the context chunk
  --max_epochs MAX_EPOCHS
  --train_path TRAIN_PATH
                        path to the folder containing train datasets
  --test_path TEST_PATH
                        path to the folder containing test datasets
  --checkpoint_path CHECKPOINT_PATH
                        ckpt path to load
  --hparams HPARAMS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --infer_one
  --save_path SAVE_PATH
                        path to save outputs
  --lr LR
  --warmup_ratio WARMUP_RATIO
  --gpus GPUS
  --num_nodes NUM_NODES
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
  --squad_ver SQUAD_VER
                        squad version 1.0 or 2.0
  --shuffle             [squad v2.0] shuffle train dataset
  --cleanse             [squad v2.0] equalize # of dataset for each label
```

## Evaluate
### Make Binary
`bash script/run_binary.sh`
```
  --hparams HPARAMS     path to hparams.yaml saved by training
  --model_binary MODEL_BINARY
                        path to .ckpt saved by training
  --output_dir OUTPUT_DIR
                        output file to save
```
* this will create `config.json` and `pytorch_model.bin` under `args.output_dir` file 

### Real Time Inference
`bash script/eval.sh`
```
  --max_len MAX_LEN     max length of the context chunk
  --batch               operate in batch
  --batch_size BATCH_SIZE
  --ckpt_path CKPT_PATH
                        path to binary file
```
* output format
  `context>`: insert context  
  `question>`: insert question  
  
* batch로 진행시 하나의 context에 대해 batch size 만큼의 질문 진행
* batch로 진행 안할 시 context에 대해 질문이 더 이상 존재하지 않으면 "exit" 작성으로 나가기

### Create predicted output
`bash script/run_predict.sh`
```
  --max_len MAX_LEN     max length of the context chunk
  --input_path INPUT_PATH
                        path to input dataset
  --ckpt_path CKPT_PATH
                        path to binary file
  --output_file OUTPUT_FILE
                        file name to save
  --output_path OUTPUT_PATH
                        path to save
```
* output_path에 output_file 이름으로 예측한 답들을 담은 .json file 만들기

### Get Score
`bash script/run_score.sh`
* first param: path to input dataset (= args.input_path during creating predicted output)
* second param: path to created predicted output (= args_output_path + args.output_file)

* 위 step에서 생성한 .json file과 답을 가지고 F1 and exact score 계산
* 실제 korquad evaluation 방식

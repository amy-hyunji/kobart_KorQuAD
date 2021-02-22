BASE_DIR=max_len_768_ver_1.0_new_data

python3 get_model_binary.py --hparams /home/hyunji/kobart_KorQuAD/$BASE_DIR/tb_logs/default/version_0/hparams.yaml --model_binary /home/hyunji/kobart_KorQuAD/$BASE_DIR/kobart_qa-model_chp/epoch=03-val_loss=0.610.ckpt

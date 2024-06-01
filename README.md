# SCARF_BER
```bash
python run_scarf.py \
  --config_dir=configs \
  --output_dir=exp \
  --train_data_path=data/small_train.csv \
  --batch_size=32 \
  --epochs=1 \
  --lr=3e-5 \
  --emb_dim=32 \
  --encoder_depth=3 \
  --model_name="scarf" \
  --corruption_rate=0.3 \
  --device="cpu" \
  --wandb_project_name="SCARF_Project" \
  --wandb_entity="urbancomp" \
  --wandb_key=WANDB_KEY
```

```bash
python get_scarf_embedding.py \
  --config_dir=configs \
  --output_dir=exp \
  --data_path=data/small_train.csv \
  --batch_size=32 \
  --epochs=1 \
  --lr=3e-5 \
  --emb_dim=32 \
  --encoder_depth=3 \
  --model_name="scarf" \
  --corruption_rate=0.3 \
  --device="cpu" \
  --embedding_save_name="train"

```

python run_mlp.py \
  --config_dir "configs" \
  --output_dir "exp" \
  --train_data_path "data/small_train.csv" \
  --batch_size 32 \
  --epochs 1 \
  --lr 0.00003 \
  --model_name "mlp" \
  --device "cpu" \
  --wandb_project_name "test" \
  --wandb_entity "urbancomp" \
  --wandb_key "your_wandb_api_key" \
  --hidden_layer 256 128 64 32 16 \
  --dropout 0.1


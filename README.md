# SCARF_BER

## Setup
**Installation:**
Create a conda environment named `scarf`:
```bash
conda create -n scarf python=3.11 
conda activate scarf
pip install -r requirements.txt
```

**WANDB API Key:**
* Write your W&B API key to `.env`, an example is in `env_example`.
* For Kaggle users, store your W&B API key as a Kaggle secret.

## Train the SCARF model
The trained scarf model will be save in  `exp/scarf.pt` if you run the following command:
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
  --wandb_project_name="SCARF_Project" \
  --wandb_entity="urbancomp" \
```

## Getting Embeddings:
The generated embeddings are saved as a NumPy array in `exp/train.npy` if you run the following command:

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
  --embedding_save_name="train"
```

## Train the MLP Classifier:
The trained mlp classifier will be save in  `exp/mlp.pt` if you run the following command:
```bash
python run_mlp.py \
  --config_dir "configs" \
  --output_dir "exp" \
  --train_data_path "data/small_train.csv" \
  --test_data_path "data/small_test.csv" \
  --batch_size 32 \
  --epochs 1 \
  --lr 0.00003 \
  --model_name "mlp" \
  --wandb_project_name "test" \
  --wandb_entity "urbancomp" \
  --hidden_layer 256 128 64 32 16 \
  --dropout 0.1
```
## Processing Data:

It's crucial to process the training set before the test set. This is because some preprocessing steps, like normalization or standardization, require calculating statistics from the training data. These statistics are then applied to the test data to ensure consistency.

* For training set:
The processed dataset will be save in  `exp/processed_train.csv` if you run the following command:
```bash
python get_processed_dataset.py \
  --config_dir "configs" \
  --output_dir "exp" \
  --data_path "data/small_train.csv" \
  --output_csv_name "processed_train" \
  --is_train
```
* For test set:
The processed dataset will be save in  `exp/processed_test.csv` if you run the following command:
```bash
python get_processed_dataset.py \
  --config_dir "configs" \
  --output_dir "exp" \
  --data_path "data/small_test.csv" \
  --output_csv_name "processed_test" \
```



import argparse
import os
import pandas as pd
import wandb
from scripts.utils import set_seed, load_from_yaml, get_features_from_yaml
from scripts.data_processor import DataProcessor
from scripts.model import SCARF
from scripts.dataloader import ScarfToDataLoader
from train import train_encoder
from dotenv import load_dotenv
# Suppress Dtype and Future warnings from pandas
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a consistent seed for reproducibility
set_seed(42)
load_dotenv()


# Argument parsing
parser = argparse.ArgumentParser(description='Train SCARF model')
parser.add_argument('--config_dir', default='configs', help='Directory for configuration files')
parser.add_argument('--output_dir', default='exp', help='Output directory for models and stats')
parser.add_argument('--train_data_path', default='data/small_train.csv', help='Data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--emb_dim', type=int, default=32, help='Dimensionality of the embedding space')
parser.add_argument('--encoder_depth', type=int, default=3, help='Depth of the encoder model')
parser.add_argument('--model_name', type=str, default="scarf", help='Name of saved model')
parser.add_argument('--corruption_rate', type=float, default=0.3, help='Rate of corruption applied during training')
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--wandb_project_name', type=str, required=True, help='Name of wandb project')
parser.add_argument('--wandb_entity', type=str, default="urbancomp", help='Name of wandb entity')
parser.add_argument('--wandb_key', type=str)
args = parser.parse_args()

# Ensure output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Load configurations from YAML files
preprocessing_config = load_from_yaml(f"{args.config_dir}/preprocess_config.yaml")
energy_config = load_from_yaml(f"{args.config_dir}/energy_config.yaml")
column_type_path = f"{args.output_dir}/column_type_classification.yaml"
train_stats_path = f"{args.output_dir}/train_stats.json"
scaler_path = f"{args.output_dir}/scaler.joblib"
encoder_path = f"{args.output_dir}/encoder.joblib"
small_area_path = f"{args.config_dir}/small_area.yaml"
target = "EnergyRating"
feature_config = load_from_yaml(f"{args.config_dir}/column_type_classification.yaml")
features = get_features_from_yaml(feature_config, target)
energyRatingEncoding = energy_config["original_rating_encoding"]


wandb_key = os.getenv('WANDB_KEY')
if not wandb_key:
    wandb_key = args.wandb_key

# Initialize Weights & Biases
wandb.login(key=wandb_key)
wandb.init(
    project="Scarf",
    name=args.wandb_project_name,
    entity=args.wandb_entity,
    config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "feature_num": len(features),
        "class_num": len(energyRatingEncoding),
        "features": features,
        "model_save_dir": args.output_dir,
        "model_name": args.model_name,
        "emb_dim": args.emb_dim,
        "encoder_depth": args.encoder_depth,
        "corruption_rate": args.corruption_rate,
    }
)

# Load datasets
df_train = pd.read_csv(f"{args.train_data_path}")

# Process datasets
processor = DataProcessor(preprocessing_config, train_stats_path, column_type_path, scaler_path, encoder_path, small_area_path, target, features)
train_df = processor.process(df_train, is_train=True)

# Initialize and train the model
model = SCARF(input_dim=len(features), emb_dim=args.emb_dim, encoder_depth=args.encoder_depth, corruption_rate=args.corruption_rate)
train_encoder(train_df, 
              ScarfToDataLoader, 
              model, 
              device=args.device, 
              target_col=target, 
              batch_size=args.batch_size, 
              lr=args.lr, 
              epochs=args.epochs, 
              model_save_dir=args.output_dir, 
              model_name=args.model_name)

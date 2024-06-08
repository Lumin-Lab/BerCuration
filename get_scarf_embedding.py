import argparse
import os
import pandas as pd
import torch
from scripts.utils import set_seed, load_from_yaml, get_features_from_yaml, load_model
from scripts.data_processor import DataProcessor
from scripts.scarf import SCARF
from scripts.dataloader import ScarfToDataLoader
# Suppress Dtype and Future warnings from pandas
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a consistent seed for reproducibility
set_seed(42)

# Argument parsing
parser = argparse.ArgumentParser(description='Generate Embeddings')
parser.add_argument('--config_dir', default='configs', help='Directory for configuration files')
parser.add_argument('--output_dir', default='exp', help='Output directory for models and stats')
parser.add_argument('--data_path', type=str, help='Specify the data path for the file you want to convert into SCARF embeddings.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--emb_dim', type=int, default=32, help='Dimensionality of the embedding space')
parser.add_argument('--encoder_depth', type=int, default=3, help='Depth of the encoder model')
parser.add_argument('--model_name', type=str, default="scarf", help='Name of saved model')
parser.add_argument('--corruption_rate', type=float, default=0.3, help='Rate of corruption applied during training')
# parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--embedding_save_name', type=str, required=True)
args = parser.parse_args()

# Ensure output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = "cuda" if  torch.cuda.is_available() else "cpu"

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


# Load datasets
df = pd.read_csv(f"{args.data_path}")

# Process datasets
# processor = DataProcessor(preprocessing_config, train_stats_path, column_type_path, scaler_path, encoder_path, small_area_path, target, features)
# data_df = processor.process(df, is_train=True)

# Initialize and train the model
model = SCARF(input_dim=len(features), emb_dim=args.emb_dim, encoder_depth=args.encoder_depth, corruption_rate=args.corruption_rate)
model = load_model(model_dir = args.output_dir, 
           model_name= args.model_name,
           model = model,
           device = device)

model.eval()
embeddings = []
dataloader = ScarfToDataLoader(df, 
                                target_col=target, 
                                batch_size=args.batch_size, 
                                shuffle=False).dataloader

with torch.no_grad():
    for f, _ in dataloader:
        embeddings.append(model.get_embeddings(f.to(device)))
        
embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.cpu()
embeddings_numpy= embeddings.numpy()
np.save(f"{args.output_dir}/{args.embedding_save_name}.npy", embeddings_numpy)
 


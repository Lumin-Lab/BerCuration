import argparse
import os
import pandas as pd
from scripts.utils import set_seed, load_from_yaml, get_features_from_yaml
from scripts.data_processor import DataProcessor
# Suppress Dtype and Future warnings from pandas
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a consistent seed for reproducibility
set_seed(42)


# Argument parsing
parser = argparse.ArgumentParser(description='Train SCARF model')
parser.add_argument('--config_dir', default='configs', help='Directory for configuration files')
parser.add_argument('--output_dir', default='exp', help='Output directory for models and stats')
parser.add_argument('--output_csv_name', required=True)
parser.add_argument('--data_path', default='data/small_train.csv', help='Path to the training data file.')
parser.add_argument('--is_train', action=argparse.BooleanOptionalAction)

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


# Load datasets
df = pd.read_csv(f"{args.data_path}")

# Process datasets
processor = DataProcessor(preprocessing_config, train_stats_path, column_type_path, scaler_path, encoder_path, small_area_path, target, features)
processed_df = processor.process(df, args.is_train)

processed_df.to_csv(f"{args.output_dir}/{args.output_csv_name}.csv", index=False)






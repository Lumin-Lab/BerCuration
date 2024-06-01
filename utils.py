import torch
import numpy as np
import random
import json
import os
# from sqlalchemy import create_engine
import pandas as pd
import yaml
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
# import psycopg2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_wandb_metrics(mode, acc, per_category_accuracies):
    metrics = {
    f"{mode}/{mode}_acc": acc,
    }

    for name, acc in per_category_accuracies.items():
       metrics[f"{mode}/{name}_acc"] = acc
    return metrics

def save_to_json(fname, data):
    """
    Save data to a JSON file.
    
    If the file already exists, the function reads the current content of the file,
    updates it with the new data, and writes it back.
    """
    try:
        with open(fname, 'r') as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = {}
    
    # Update the existing data with the new data
    existing_data.update(data)
    
    with open(fname, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

def load_from_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return data

# def load_df_from_db():
#     POSTGRES_USER = os.environ.get('POSTGRES_USER')
#     POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')
#     POSTGRES_DB = os.environ.get('POSTGRES_DB')
#     POSTGRES_PORT = os.environ.get('POSTGRES_PORT')
#     POSTGRES_HOST = os.environ.get('POSTGRES_HOST')
#     POSTGRES_SCHEMA = os.environ.get('POSTGRES_SCHEMA')
#     DATABASE_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

#     print("Database_url is ", DATABASE_URL)

#     # Create an engine
#     engine = create_engine(DATABASE_URL)


#     # Connect to the database
#     connection = engine.connect()

#     data = pd.concat(pd.read_sql(f"SELECT * FROM {POSTGRES_SCHEMA}.ber", connection, chunksize=50000))
    
#     return data

def prune_and_compute(chunk, areas, min_budget, max_budget):
    valid_combinations_chunk = []
    
    for d_idx, d in enumerate(chunk[0]):
        for w_idx, w in enumerate(chunk[1]):
            for f_idx, f in enumerate(chunk[2]):
                for r_idx, r in enumerate(chunk[3]):
                    # Prune here: if the current combination of 4 products exceeds max or is below min budget, no need to loop over wall prices
                    partial_sum = np.dot([d, w, f, r], areas[:-1])
                    if partial_sum > max_budget or partial_sum < min_budget:
                        continue

                    for wa_idx, wa in enumerate(chunk[4]):
                        total_cost = partial_sum + wa * areas[-1]
                        if min_budget <= total_cost <= max_budget:
                            valid_combinations_chunk.append({
                                'door_id': d_idx,
                                'window_id': w_idx,
                                'floor_id': f_idx,
                                'roof_id': r_idx,
                                'wall_id': wa_idx,
                                'cost': total_cost
                            })
                            
    return valid_combinations_chunk

def save_as_yaml(data_dict, file_name):
    # Serialize the dictionary to a YAML formatted string
    yaml_str = yaml.dump(data_dict)
    # Save the string to a file
    with open(file_name, 'w') as yaml_file:
        yaml_file.write(yaml_str)


def load_from_yaml(file_name):
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    return config

# def plot_confusion_matrix(cm, energyRatingEncoding):
#     """
#     Visualizes the confusion matrix using a heatmap.
    
#     Args:
#     - cm (numpy.array): The confusion matrix.
#     - energyRatingEncoding (dict): Dictionary with encoding for the energy ratings.
    
#     Returns:
#     None, but displays a heatmap of the confusion matrix.
#     """
    
#     plt.figure(figsize=(10,7))
    
#     # Define the class names using keys from the energyRatingEncoding dictionary
#     class_names = energyRatingEncoding.keys()
    
#     # Plot the heatmap
#     sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
#                 xticklabels=class_names, 
#                 yticklabels=class_names)
    
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
    
def convert_ber_to_energy_rating(ber_value, ber_categories):
    prev_threshold = 0
    for i, category in enumerate(ber_categories):
        if category['ber'] == '≤ 25':
            if ber_value <= 25:
                return category['category']
            prev_threshold = 25
        else:
            threshold = int(category['ber'][2:])
            # if ber_value is in between the previous and the current threshold, return the previous category.
            if prev_threshold < ber_value <= threshold:
                return ber_categories[i - 1]['category']
            prev_threshold = threshold
            
    return ber_categories[-1]['category'] if ber_value > threshold else 'Unknown'


def get_features_from_yaml(feature_config, target):
    # feature_config = load_from_yaml(dl_config["col_type_path"])
    cat_cols = feature_config['cat_cols'] if feature_config['cat_cols'] is not None else []
    nmr_cols = feature_config['nmr_cols'] if feature_config['nmr_cols'] is not None else []
    features = cat_cols + nmr_cols
    features = [f for f in features if f!= target]
    return features

def merge_energy_class(df, merge_config):
    # df_copy = df.copy()
    df["EnergyRating"] = df["EnergyRating"].str.strip()
    df["EnergyRating"] = df["EnergyRating"].map(merge_config)
    return df


def compute_metrics(outputs, labels, energyRatingEncoding):
    """Compute overall and per-category accuracy."""
    if len(outputs.shape) == 2:
        _, predictions = torch.max(outputs, 1)
    else:
        predictions = outputs
    overall_correct = (predictions == labels).sum().item()
    overall_accuracy = overall_correct / labels.size(0)
    per_category_accuracy = {}
    for class_name, class_label in energyRatingEncoding.items():
        correct = ((predictions == class_label) & (labels == class_label)).sum().item()
        total = (labels == class_label).sum().item()
        accuracy = correct / total if total != 0 else 0
        per_category_accuracy[class_name] = accuracy
    
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    f1 = f1_score(labels, predictions, average='macro')
    return overall_accuracy, per_category_accuracy, f1, predictions, labels

def save_model(model, model_dir, model_name):
    """
    Save the model's state_dict and training configuration.
    
    Args:
    - model (nn.Module): the trained model.
    - config (dict): the training configuration/settings.
    - path (str): path to save the model checkpoint.
    """
    
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    
    path = f"{model_dir}/{model_name}.pt"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(checkpoint, path)
    print(f"Model saved at {path}")
    

def load_model(model_dir, model_name, model, device):
    """
    Load the saved model checkpoint.
    
    Args:
    - path (str): path to the saved model checkpoint.
    - device (str): device to which the model is loaded.
    
    Returns:
    - model (nn.Module): the loaded model.
    - config (dict): the training configuration/settings used for this model.
    """
    path = f"{model_dir}/{model_name}.pt"

    checkpoint = torch.load(path, map_location=device)
    
    # Load the model weights
    model.load_state_dict(checkpoint["state_dict"])
    
    # Move the model to the desired device
    model.to(device)

    print(f"Model loaded from {path}")
    
    return model

def convert_energy_rating_to_number(df, energy_rating_encoding):
    # df_copy = df.copy()
    df['EnergyRating'] = df['EnergyRating'].str.strip()
    df['EnergyRating'] = df['EnergyRating'].map(energy_rating_encoding)
    return df

def convert_number_to_energy_ratings(lst, energy_rating_encoding):
    reversed_encoding = {v: k for k, v in energy_rating_encoding.items()}
    ratings = [reversed_encoding[v] for v in lst]
    return ratings



# def count_labels(input_number, energy_rating_merge, energy_rating_encoding):
#     # Reverse the energy_rating_merge to get letters for each number
#     reverse_merge = {}
#     for key, value in energy_rating_merge.items():
#         reverse_merge.setdefault(value, []).append(key)

#     # Reverse the energy_rating_encoding to get numbers for each letter
#     reverse_encoding = {v: k for k, v in energy_rating_encoding.items()}

#     # Find the corresponding letter for the input number
#     letter = reverse_encoding.get(input_number)

#     if letter is not None:
#         # Count the occurrences of the letter in the reverse_merge
#         return len(reverse_merge.get(letter, []))
#     else:
#         return 0  # Or an appropriate value/error if the number is not found

# def fetch_user_ber_record(dbname, user, password, host, port, schema, berID, mapping):
    
#     # Connect to the PostgreSQL database
#     conn = psycopg2.connect(
#         dbname=dbname,
#         user=user,
#         password=password,
#         host=host,
#         port=port
#     )

#     # Create a cursor object
#     cur = conn.cursor()

#     # Execute a SQL query to fetch the row
#     cur.execute(f"SELECT * FROM {schema}.ber OFFSET {berID}-1 LIMIT 1;")

#     # Fetch the result
#     row = cur.fetchone()

#     # Close the cursor and connection
#     cur.close()
#     conn.close()

#     # If a row was fetched
#     if row:
#         # Get column names from the database table
#         column_names = [desc[0] for desc in cur.description]
        
#         # Create a DataFrame
#         df = pd.DataFrame([row], columns=column_names)

#         df = convert_column_names(mapping, df)

#     else:
#         raise FileNotFoundError("No row fetched.")

#     return df

def convert_column_names(column_name_mapping, df):
    return df.rename(columns=column_name_mapping)
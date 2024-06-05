from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
import warnings
from scripts.utils import save_to_json, load_from_json, save_as_yaml, load_from_yaml, get_features_from_yaml
from joblib import dump, load


# Ignore the SettingWithCopyWarning from pandas
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def split_data_into_train_dev_test(X, y, config):
    """
    Split the data into training, development, and testing sets. The function then concatenates 
    the features and labels of each set, returning them as separate DataFrames.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Features of the dataset.
    - y (pd.Series or np.ndarray): Target labels of the dataset.
    - config (dict): Configuration dictionary containing train and test split sizes.

    Returns:
    - tuple: Contains three DataFrames for the train, development, and test sets respectively.
             Each DataFrame contains both features and labels concatenated.
    """
    
    # Compute the size for the development set. It's the remaining data after allocating
    # for training and testing based on the provided proportions in the config.
    dev_size = 1 - config["training"]["train_size"] - config["training"]["test_size"]

    # Initially split the data into a training set and a temporary set (which will further be split 
    # into development and testing sets).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=config["training"]["train_size"], stratify=y
    )
    
    # Split the temporary dataset into development and test sets based on the calculated development size.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, train_size=dev_size / (config["training"]["test_size"] + dev_size), stratify=y_temp
    )

    # Create final DataFrames for train, development, and test sets by concatenating features and labels.
    train_set = pd.concat([X_train, y_train], axis=1)
    dev_set = pd.concat([X_dev, y_dev], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    return train_set, dev_set, test_set


def adjust_year_of_construction(df):
    """
    Adjusts the "Year_of_Construction" column values based on the given conditions.
    """
    df.loc[df["Year_of_Construction"] < 1900, "Year_of_Construction"] = 1900
    df.loc[df["Year_of_Construction"] > 2020, "Year_of_Construction"] = 2020
    
    return df


def aggregate_to_mean(df, reference_cols, target_col):
    """
    Group the DataFrame by the reference columns and calculate the mean of the target column.
    The results are returned as a dictionary mapping from tuples of reference column values to the mean.
    """
    grouped = df.groupby(reference_cols)[target_col].mean().to_dict()
    overall_mean = df[target_col].mean()
    return grouped, overall_mean
 

def fill_based_on_avg(df, target_col, 
                      is_train,
                      train_stats_path,
                      reference_cols=None, 
                      lower_bound=None, 
                      upper_bound=None):
    """
    Fill values in the target column based on conditions. 
    """
    mask = df[target_col].isna()
    if lower_bound is not None:
        mask |= df[target_col] <= lower_bound
    if upper_bound is not None:
        mask |= df[target_col] >= upper_bound

    if is_train:
        grouped, overall_mean = aggregate_to_mean(df, reference_cols, target_col)
        save_to_json(train_stats_path, 
                     {target_col: 
                      {"grouped": grouped,
                      "overall_mean": overall_mean}}
                     )
    else:
        data = load_from_json(train_stats_path)
        grouped = data[target_col]["grouped"]
        overall_mean =  data[target_col]["overall_mean"]

    if reference_cols:
        df.loc[mask, target_col] = df.loc[mask, reference_cols].apply(tuple, axis=1).map(grouped).fillna(overall_mean)
    else:
        df.loc[mask, target_col] = overall_mean

    return df

def aggregate_to_freq(df, reference_cols, target_col):
    """
    Group the DataFrame by the reference columns and get the most frequent category of the target column.
    The results are returned as a dictionary mapping from tuples of reference column values to the mode.
    """
    mode_grouped = df.groupby(reference_cols)[target_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    overall_mode = df[target_col].mode().iloc[0] if not df[target_col].mode().empty else None
    return mode_grouped.to_dict(), overall_mode


def fill_based_on_freq(df, target_col, 
                       is_train,
                       train_stats_path,
                       reference_cols=None, 
                       threshold=0.0001):
    """
    Fill NaN values or less frequent values in the target column based on reference columns.
    """
    rare_categories = df[target_col].value_counts(normalize=True).loc[lambda x: x < threshold].index.tolist()
    
    if is_train:
        grouped, overall_mode = aggregate_to_freq(df, reference_cols, target_col)
        save_to_json(train_stats_path, 
                        {target_col: 
                        {"grouped": grouped,
                        "overall_mode": overall_mode}}
                        )
    else:
        data = load_from_json(train_stats_path)
        grouped = data[target_col]["grouped"]
        overall_mode =  data[target_col]["overall_mode"]

    if reference_cols:
        mask = df[target_col].isna() | df[target_col].isin(rare_categories)
        
        df.loc[mask, target_col] = df.loc[mask, reference_cols].apply(tuple, axis=1).map(grouped).fillna(overall_mode)
    else:
        mask = df[target_col].isna() | df[target_col].isin(rare_categories)
        df.loc[mask, target_col] = overall_mode

    return df



def compute_floor_area(df):
    """
    Compute the FloorArea by summing up various area columns.
    """
    df['FloorArea'] = (df.GroundFloorArea_sq_m_ + 
                       df.FirstFloorArea + 
                       df.SecondFloorArea + 
                       df.ThirdFloorArea + 
                       df.RoomInRoofArea)
    return df



def mapping_values(df, mapping, column_name):
    """
    Replace values in the specified column based on a given mapping.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - mapping (dict): Mapping of original values to new values.
    - column_name (str): Name of the column in which values should be replaced. 

    Returns:
    - pd.DataFrame: DataFrame with replaced values in the specified column.
    """
    df[column_name] = df[column_name].replace(mapping)
    return df

def strip_categorical_columns(df):
    """
    Applies str.strip() to all categorical columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with stripped string values in all categorical columns.
    """
    # Identify categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    
    # Apply str.strip() to each categorical column
    for col in cat_columns:
        df[col] = df[col].str.strip()

    return df

def adjust_primary_circuit_loss(df):
    """
    Modifies the 'PrimaryCircuitLoss' based on certain conditions related to 'CylinderStat'.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with modified 'PrimaryCircuitLoss' values based on the given conditions.
    """
    mask_uninsulated_with_cylinder_stat = (df['PrimaryCircuitLoss'] == "Boiler with uninsulated primar") & (df['CylinderStat'] == "YES")
    mask_insulated_with_cylinder_stat = (df['PrimaryCircuitLoss'] == "Boiler with insulated primary") & (df['CylinderStat'] == "YES")

    df.loc[mask_uninsulated_with_cylinder_stat, 'PrimaryCircuitLoss'] = 610
    df.loc[mask_insulated_with_cylinder_stat, 'PrimaryCircuitLoss'] = 360

    return df

def preprocess_area(df, preprocessing_config, train_stats_path, is_train=False):
    data = df.copy()
    # Handle missing/invalid data using average values
    data = compute_floor_area(data)
    for col in preprocessing_config["imputation_area"]["average"]:
        base_cols = ["DwellingTypeDescr"]
        data = fill_based_on_avg(data, col, is_train, train_stats_path, base_cols, lower_bound=0)
    return data



def preprocess_data(df, preprocessing_config, train_stats_path, is_train):
    """
    Preprocesses the input data based on specified transformations.
    
    Steps include:
    1. Trimming whitespace from categorical columns.
    2. Correcting the 'Year_of_Construction' column.
    3. Applying predefined mappings to columns.
    4. Computing total floor area.
    5. Handling missing or invalid data with average values based on dwelling type.
    6. Handling missing or invalid data with frequent values based on dwelling type.
    7. Setting NaN values in 'ElectricityConsumption' to 0.
    8. Adjusting 'PrimaryCircuitLoss' based on 'CylinderStat' conditions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to preprocess.
    - preprocessing_config (dict): Configuration for preprocessing steps.
    - train_stats_path (str): Path to training statistics.
    - is_train (bool): Indicates whether the dataset is for training.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Create a copy to avoid altering the original data
    data = df.copy()

    # Strip whitespace from categorical columns
    data = strip_categorical_columns(data)

    # Adjust years outside the valid range
    data = adjust_year_of_construction(data)

    # Calculate total floor area
    data = compute_floor_area(data)

    # Apply predefined mappings
    for feature, alias_feature in preprocessing_config['mappings']["mapping_categories"].items():
        if isinstance(alias_feature, str):
            data = mapping_values(data, preprocessing_config['mappings'][alias_feature], column_name=feature)
        else:
            for f in alias_feature:
               data = mapping_values(data, preprocessing_config['mappings'][f], column_name=feature)

    # Handle missing/invalid data using frequent values
    for col in preprocessing_config["imputation"]["most_frequent"]:
        base_cols = ["CountyName"] if col == "SA_Code" else ["DwellingTypeDescr"]
        data = fill_based_on_freq(data, col, is_train, train_stats_path, base_cols)

    # Handle missing/invalid data using average values
    for col in preprocessing_config["imputation"]["average"]:
        if col == "HSMainSystemEfficiency":
            base_cols = ["MainSpaceHeatingFuel"]
        elif col == "WHMainSystemEff":
            base_cols = ["MainWaterHeatingFuel"]
        elif col == "InsulationThickness":
            base_cols = ["InsulationType"]
        else:
            base_cols = ["DwellingTypeDescr"]

        data = fill_based_on_avg(data, col, is_train, train_stats_path, base_cols, lower_bound=0 if col != "InsulationThickness" else None)

    # Handle 'ElectricityConsumption' NaNs
    data.ElectricityConsumption.fillna(0, inplace=True)

    # Adjust 'PrimaryCircuitLoss' values
    data = adjust_primary_circuit_loss(data)

    return data

def preprocess_inference(df, preprocessing_config, train_stats_path, is_train=False):
    """
    Preprocesses the input data based on specified transformations.
    
    Steps include:
    1. Trimming whitespace from categorical columns.
    2. Correcting the 'Year_of_Construction' column.
    3. Applying predefined mappings to columns.
    5. Handling missing or invalid data with average values based on dwelling type.
    6. Handling missing or invalid data with frequent values based on dwelling type.
    7. Setting NaN values in 'ElectricityConsumption' to 0.
    8. Adjusting 'PrimaryCircuitLoss' based on 'CylinderStat' conditions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to preprocess.
    - preprocessing_config (dict): Configuration for preprocessing steps.
    - train_stats_path (str): Path to training statistics.
    - is_train (bool): Indicates whether the dataset is for training.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Create a copy to avoid altering the original data
    data = df.copy()

    # Strip whitespace from categorical columns
    data = strip_categorical_columns(data)

    # Adjust years outside the valid range
    data = adjust_year_of_construction(data)


    # Apply predefined mappings
    for feature, alias_feature in preprocessing_config['mappings']["mapping_categories"].items():
        if isinstance(alias_feature, str):
            data = mapping_values(data, preprocessing_config['mappings'][alias_feature], column_name=feature)
        else:
            for f in alias_feature:
               data = mapping_values(data, preprocessing_config['mappings'][f], column_name=feature)

    # Handle missing/invalid data using frequent values
    for col in preprocessing_config["imputation"]["most_frequent"]:
        base_cols = ["CountyName"] if col == "SA_Code" else ["DwellingTypeDescr"]
        data = fill_based_on_freq(data, col, is_train, train_stats_path, base_cols)

    # Handle missing/invalid data using average values
    for col in preprocessing_config["imputation"]["average"]:
        if col == "HSMainSystemEfficiency":
            base_cols = ["MainSpaceHeatingFuel"]
        elif col == "WHMainSystemEff":
            base_cols = ["MainWaterHeatingFuel"]
        elif col == "InsulationThickness":
            base_cols = ["InsulationType"]
        else:
            base_cols = ["DwellingTypeDescr"]
        if col not in preprocessing_config["imputation_area"]["average"]:
            data = fill_based_on_avg(data, col, is_train, train_stats_path, base_cols, lower_bound=0 if col != "InsulationThickness" else None)

    # Handle 'ElectricityConsumption' NaNs
    data.ElectricityConsumption.fillna(0, inplace=True)

    # Adjust 'PrimaryCircuitLoss' values
    data = adjust_primary_circuit_loss(data)

    return data


def split_cat_nmr_cols(df, is_train, col_path = 'dl_configs/column_type.yaml'):
    if is_train:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        nmr_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        col_dict = {'cat_cols': cat_cols, 'nmr_cols': nmr_cols}
        save_as_yaml(col_dict, col_path)
    else:
        col_dict = load_from_yaml(col_path)
        cat_cols, nmr_cols = col_dict['cat_cols'], col_dict['nmr_cols']
    
    return cat_cols, nmr_cols

def scale(df, nmr_cols, is_train, scaler_path = "dl_configs/scaler.joblib"):
    if is_train:
        # scaler = QuantileTransformer(output_distribution='normal', 
        #                              n_quantiles=max(min(len(df) // 30, 1000), 10),
        #                              subsample=int(1e9),
        #                              random_state=42)
        scaler = StandardScaler()
        scaler.fit(df[nmr_cols])
        dump(scaler, scaler_path)
    else:
        scaler = load(scaler_path)
    df[nmr_cols] = scaler.transform(df[nmr_cols])
    return df

def cat_encode(df, cat_cols, is_train, encoder_path="dl_configs/encoder.joblib", 
               small_area_path = "dl_configs/small_area.yaml"):
    encoders = {}

    if is_train:
        # ber = pd.read_csv("data/ber.csv")
        # If is_train is True, fit a new encoder for each categorical column and save them.
        for col in cat_cols:
            encoder = LabelEncoder()
            if col == "SA_Code":
                encoder.fit(load_from_yaml(small_area_path))
            else:
                encoder.fit(df[col])
            encoders[col] = encoder
            df[col] = encoder.transform(df[col])
            
        
        # Save the encoders to a file
        dump(encoders, encoder_path)
        
    else:
        # If is_train is False, load the encoders and use them to transform the categorical columns.
        encoders = load(encoder_path)
    
        for col in cat_cols:
            if col == "SA_Code":
                df[col] = df[col].astype(str)
            try:
                df[col] = encoders[col].transform(df[col])
            except:
                encoder.fit(df[col])
                df[col] = encoder.transform(df[col])
    
    return df


class DataProcessor:
    """
    A class responsible for preprocessing data for a machine learning model.
    """

    def __init__(self, 
                 preprocessing_config, 
                 train_stats_path,
                 column_type_path,
                 scaler_path,
                 encoder_path,
                 small_area_path,
                 target,
                 features):
        """
        Initializes the DataProcessor.

        Args:
        - paths (dict): Configurations for paths.
        - model_config (dict): Configuration for model.
        """
        # self.preprocessing_config = load_from_yaml(paths["config"]["preprocessing"])
        
        # self.model_config = model_config
        # self.train_stats_path = paths["stats"]["train_stats"]
        # self.column_type_path = model_config["col_type_path"]
        # self.scaler_path = model_config["scaler"]
        # self.encoder_path = model_config["encoder"]
        # self.small_area_path = paths['small_area']['small_area']
        # self.target = model_config["model_info"]["target"]
        # self.features = get_features_from_yaml(model_config)
        self.preprocessing_config = preprocessing_config
        self.train_stats_path = train_stats_path
        self.column_type_path = column_type_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.small_area_path = small_area_path
        self.target = target
        self.features = features


    def process(self, df, is_train, real_time_inference=False):
        """
        Preprocesses the given dataset.

        Args:
        - df (pd.DataFrame): Dataset to be preprocessed.
        - is_train (bool): Indicates if the dataset is for training.

        Returns:
        - pd.DataFrame: Preprocessed dataset.
        """
        data = df.copy()

        if not real_time_inference:
            # Apply preprocessing steps
            data = self._preprocess_data(data, is_train)
        else:
            data = self._preprocess_inference(data)

        # Filter columns as per model configuration
        data = data[self.features + [self.target]]

        # Handle categorical columns encoding
        cat_cols, _ = self._split_cat_nmr_cols(data, is_train, self.column_type_path)

        data = self._cat_encode(data, cat_cols, is_train, self.encoder_path, self.small_area_path)

        # Handle scaling for numerical columns
        cols = data.columns.tolist()
        # _, nmr_cols = self._split_cat_nmr_cols(data, is_train=False)
        cols.remove(self.target)
        data = self._scale(data, cols, is_train, self.scaler_path)

        return data.iloc[-len(df):] if not is_train else data

    # def get_energy_rating_encoding(self):
    #     """Returns the encoding used for energy ratings."""
    #     return self.energy_rating_encoding

    def _preprocess_data(self, data, is_train):
        """Applies preprocessing configurations."""
        return preprocess_data(data, self.preprocessing_config, self.train_stats_path, is_train)

    def _split_cat_nmr_cols(self, data, is_train, col_type_path):
        """Divides columns into categorical and numerical types."""
        return split_cat_nmr_cols(data, is_train, col_type_path)

    def _cat_encode(self, data, cat_cols, is_train, encoder_path, small_area_path):
        """Encodes categorical columns."""
        return cat_encode(data, cat_cols, is_train, encoder_path, small_area_path)

    def _scale(self, data, nmr_cols, is_train, scaler_path):
        """Scales numerical columns."""
        return scale(data, nmr_cols, is_train, scaler_path)

    def _preprocess_inference(self, data):
        return preprocess_inference(data, self.preprocessing_config, self.train_stats_path)

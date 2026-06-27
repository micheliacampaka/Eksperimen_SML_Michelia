import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_data(
    input_path="adult.csv",
    output_folder="preprocessing"
):
    
    df = pd.read_csv(input_path)

  
    df = df.drop_duplicates()

   
    X = df.drop("income", axis=1)
    y = df["income"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
   
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

   
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])


    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    
    processed_df = pd.DataFrame(
        X_processed.toarray()
        if hasattr(X_processed, "toarray")
        else X_processed,
        columns=feature_names
    )

    processed_df["income"] = y

    
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        "adult_preprocessed.csv"
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    print("Preprocessing selesai.")
    print(f"Dataset disimpan pada: {output_path}")


if __name__ == "__main__":
    preprocess_data()
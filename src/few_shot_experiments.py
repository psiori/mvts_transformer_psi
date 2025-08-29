import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np

def write_df_to_tsfile(df, path, problem_name="dataset", label_col=None, timestamps=False):
    """
    Minimal .ts file writer (sktime-compatible).

    Parameters
    ----------
    df : pandas.DataFrame
        Full DataFrame containing time-series columns and optionally a label column.
        Time-series cells: list, pd.Series, or 1D np.ndarray of numbers.
    path : str
        Destination .ts file path.
    problem_name : str
        Name in @problemName in the header.
    label_col : str or None
        Name of the column holding class labels.
        If None, no labels are included.
    timestamps : bool
        If True, write (time,value) pairs; if False, write only values.
    """
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
        class_labels = df[label_col].tolist()
        ts_df = df.drop(columns=[label_col])
    else:
        class_labels = None
        ts_df = df

    n_cases, n_dims = ts_df.shape

    with open(path, "w", encoding="utf-8") as f:
        # ---- Header ----
        f.write(f"@problemName {problem_name}\n")
        f.write(f"@timeStamps {'true' if timestamps else 'false'}\n")
        f.write(f"@univariate {'true' if n_dims == 1 else 'false'}\n")
        f.write(f"@classLabel {'true' if class_labels is not None else 'false'}")
        if class_labels is not None:
            # Header lists all possible distinct labels sorted
            f.write(" " + " ".join(map(str, sorted(set(class_labels)))))
        f.write("\n@data\n")

        # ---- Data section ----
        for i in range(n_cases):
            row_parts = []
            for d in range(n_dims):
                vals = ts_df.iloc[i, d]
                if isinstance(vals, (pd.Series, np.ndarray)):
                    vals = vals.tolist()
                elif not isinstance(vals, list):
                    vals = list(vals)
                if timestamps:
                    dim_str = ",".join(f"({t},{v})" for t, v in enumerate(vals))
                else:
                    dim_str = ",".join(map(str, vals))
                row_parts.append(dim_str)

            row_str = ":".join(row_parts)
            if class_labels is not None:
                row_str += f":{class_labels[i]}"
            f.write(row_str + "\n")

    print(f".ts file written to {path}")


def build_random_few_shot_dataset(data_path=None, few_shot_data_path = None, no_positive = 1, no_negative = 1):

    df = load_from_tsfile_to_dataframe(data_path, return_separate_X_and_y=False)

    positive_samples = df[df['class_vals'] == 'normal'].sample(no_positive, replace=False)
    negative_samples = df[df['class_vals'] == 'abnormal'].sample(no_negative, replace=False)
    df_train = pd.concat([positive_samples, negative_samples])
    df_test = df.drop(df_train.index)

    write_df_to_tsfile(df_train,few_shot_data_path + 'TRAIN.ts' , timestamps=False, label_col='class_vals')
    write_df_to_tsfile(df_test,few_shot_data_path + 'TEST.ts', timestamps=False, label_col='class_vals')

    return df_train, df_test

def build_all_few_shot_datasets()
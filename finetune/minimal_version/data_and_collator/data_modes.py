def get_data_modes(type='h5'):
    """
    Return data loading configurations for training and validation datasets based on the input type.

    This function defines how the datasets should be loaded and what type of collator to use, depending
    on the dataset format (e.g., HDF5 or Parquet). It is typically used to configure data pipelines
    for Hugging Face + Ray workflows, where different formats and collators are supported.

    Args:
        type (str): One of 'h5', 'parquet', or 'parquet_h5'. Defaults to 'h5'.
            - 'h5': Use HDF5 format for both training and validation.
            - 'parquet': Use Parquet format for both training and validation.
            - 'parquet_h5': Use Parquet for training and HDF5 for validation.

    Returns:
        dict: Dictionary specifying the type and collator for 'train' and 'val' datasets.
    """
    # Option 1: Both training and validation use HDF5 format with streaming collators
    if type == 'h5':
        h5_data = {
            "train": {"type": "h5",
                      "collator": "streaming",
                      "load_in_trainer": False
                      },
            "val": {"type": "h5",
                    "collator": "streaming",
                    "load_in_trainer": True
                    }
            }
    # Option 2: Both training and validation use Parquet format with parquet collators
    elif type == 'parquet':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      "load_in_trainer": False
                      },
            "val": {"type": "parquet",
                    "collator": "parquet",
                    "load_in_trainer": True
                    }
            }
            
    # Option 3: Train with Parquet, validate with HDF5 (hybrid mode)
    elif type == 'parquet_h5':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      "load_in_trainer": False
                      },
            "val": {"type": "h5",
                    "collator": "streaming",
                    "load_in_trainer": True
                    }
            }

    return h5_data


# h5_data = { "train": {"type": "parquet", "collator": "parquet", "load_in_trainer": False}, "val": {"type": "parquet", "collator": "parquet", "load_in_trainer": True}}
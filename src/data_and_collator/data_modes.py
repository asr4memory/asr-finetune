def get_data_modes(type='h5'):
    """
    Return data loading configurations for training and validation datasets based on the input type.

    This function defines how the datasets should be loaded and what type of collator to use, depending
    on the dataset format (e.g., HDF5 or Parquet). It is typically used to configure data pipelines
    for Hugging Face + Ray workflows, where different formats and collators are supported.

    Args:
        type (str): Data-mode key. Defaults to 'h5'. Supported values:
            - 'h5': HDF5 (streaming collator) for both train and validation.
            - 'parquet': pre-materialized Parquet for both train and validation.
            - 'parquet_h5': Parquet for training, HDF5 for validation (hybrid).
            - 'train_parquet': only a Parquet training split.
            - 'val_parquet': only a Parquet validation split.
            - 'val_h5': only an HDF5 (streaming) validation split.

    Returns:
        dict: Dictionary specifying the type and collator for 'train' and 'val' datasets.
    """
    # Option 1: Both training and validation use HDF5 format with streaming collators
    if type == 'h5':
        h5_data = {
            "train": {"type": "h5",
                      "collator": "streaming"
                      },
            "val": {"type": "h5",
                    "collator": "streaming"
                    }
            }
    # Option 2: Both training and validation use Parquet format with parquet collators
    elif type == 'parquet':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      },
            "val": {"type": "parquet",
                    "collator": "parquet"
                    }
            }
            
    # Option 3: Train with Parquet, validate with HDF5 (hybrid mode)
    elif type == 'parquet_h5':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      },
            "val": {"type": "h5",
                    "collator": "streaming"
                    }
            }
            
    elif type == 'train_parquet':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      }
            }
            
    elif type == 'val_parquet':

        h5_data = {
            "val": {"type": "parquet",
                    "collator": "parquet"
                    }
            }
    
    elif type == 'val_h5':

        h5_data = {
            "val": {"type": "h5",
                    "collator": "streaming"
                    }
            }

    return h5_data

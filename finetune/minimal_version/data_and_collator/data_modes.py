
def get_data_modes(type='h5'):

    if type == 'h5':

        h5_data = {
            "train": {"type": "h5",
                      "collator": "streaming"
                      },
            "val": {"type": "h5",
                    "collator": "streaming"
                    }
            }

    elif type == 'parquet':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      },
            "val": {"type": "parquet",
                    "collator": "parquet"
                    }
            }

    elif type == 'parquet_h5':

        h5_data = {
            "train": {"type": "parquet",
                      "collator": "parquet",
                      },
            "val": {"type": "h5",
                    "collator": "streaming"
                    }
            }

    return h5_data


# h5_data = {
#     "train": {"type": "h5",
#               "collator": "streaming"
#               },
#     "val": {"type": "h5",
#             "collator": "streaming"
#             }
# }
#
# parquet_data = {
#     "train": {"type": "parquet",
#               "collator": "parquet",
#               },
#     "val": {"type": "parquet",
#             "collator": "parquet"
#             }
# }
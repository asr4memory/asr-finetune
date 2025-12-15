import pdb
import os
from models import get_whisper_models_local as get_models
from utils import create_ray_indexloaders, MultiShardStreamingCollator, MultiShardStreamingCollator2

def main():
    model, feature_extractor, tokenizer, processor = get_models("openai/whisper-large-v3", 'de')

    path_to_data = r'/data_example/datasets/'
    dataset_name = 'eg_dataset_complete_v2'
    batch_size = 8
    prefetch_batch = 8

    val_loader, shard_to_file_val = create_ray_indexloaders(
        os.path.join(path_to_data, dataset_name, 'val'),
        batch_size=batch_size
    )

    val_collator = MultiShardStreamingCollator2(
        shard_to_file_val,
        processor,
        feature_extractor,
        tokenizer,
        num_workers=8
    )

    val_ds_iterable = val_loader.iter_torch_batches(
        prefetch_batches=prefetch_batch,
        batch_size=batch_size,
        collate_fn=val_collator
    )

    for sample in val_ds_iterable:
        print(sample['input_features'].shape)
        # pdb.set_trace()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # Optional: ensures consistency
    main()

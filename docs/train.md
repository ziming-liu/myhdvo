# training 

NOTE: check the annotation path and data path before running. i.e. `annfile_root` and `data_root` in config file. 

To train a deep model with this codebase, you need to write a config file, then run with the command like 

```
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpu   \
    tools/train.py $config_file.py \
      --launcher pytorch  --validate($optional)

```

The config file can refer to existing configs under `hdvo/configs/`, containing `dataset`, `model`, `optimization_strategy`.


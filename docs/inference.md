# Inference 

NOTE: check the annotation path and data path before running. i.e. `annfile_root` and `data_root` in config file. 

## command 

Inference and evaluate the results with this command. 

```

torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpu --master_port=12860 \
  tools/test.py $config_file.py \
   $pretrained_checkpoint_file.pth   \
    --launcher pytorch   --save_depth($optional)
```


## test FLOPS

Install [`torchsummaryX`](https://github.com/nmhkahn/torchsummaryX) firstly. 

```

torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test_flops_counter.py $config_file.py \
   $checkpoint_file.pth   \
    --launcher pytorch --input_size 544,960($image_size)

```

## deploy 

Usually, pytorch model can be converted to TensorRT model to realize speed-up inference. 

Install [`torch2trt`](https://github.com/NVIDIA-AI-IOT/torch2trt) firstly. 

A script to TensorRT converge is 

```

python tools/deploy.py configs/hdvo/stereohdvo_posesup_s1_kittiodom.py \
   $checkpoint.pth   \
    --launcher none 
 
```
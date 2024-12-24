import os.path as osp
import pickle
import shutil
import tempfile
import random
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import time 


class TicToc():
    def __init__(self,):
       self.start = None
       self.tic()
    def tic(self, ):
       self.start = time.time()
    def toc(self, ):
        #torch.cuda.synchronize()
        end = time.time()
        assert self.start !=None, "didn't start a timer"
        elapsed_seconds = end - self.start
        return elapsed_seconds *1000 # ms


def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    #results = []
    results = [[] for _ in range(10)] # depth, pose, seq_dir
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        num_output = len(result)
        for k in range(num_output):
            results[k].extend(result[k])
        #results.extend(result)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=True, tem_dir=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        task: the tasks to solve, [stereo_depth, pose,]
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: True

    Returns:
        two level list: The prediction results. [[],[],...]
    """
    model.eval()
    num_output = 1
    results = [[] for _ in range(10)] # depth, pose, seq_dir
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    tictoc = TicToc()
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        num_output = len(result)
        """ 
        if num_output ==1:
            results[0].extend(result[0]) # depth or pose
        if num_output ==2:
            results[0].extend(result[0]) #  pose
            results[1].extend(result[1]) # seq_dir
        if num_output ==3:
            results[0].extend(result[0]) # depth 
            results[1].extend(result[1]) # GT depth
            results[2].extend(result[2]) # seq_dir
        if num_output ==4:
            results[0].extend(result[0]) # depth
            results[1].extend(result[1]) # GT depth
            results[2].extend(result[2]) # pose
            results[3].extend(result[3]) # seq_dir
        """
        for k in range(num_output):
            results[k].extend(result[k])
        if rank == 0:
            # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size * world_size):
                prog_bar.update()
    print(f"Time for testing: {tictoc.toc()} ms")
    print("time per frame: ", tictoc.toc()/len(dataset))
    # collect results from all ranks
    output_results = [[] for _ in range(num_output)]
    if gpu_collect:
        for i in range(num_output):
            output_results[i] = collect_results_gpu(results[i], len(dataset))
        #results[1] = collect_results_gpu(results[1], len(dataset))
    else:
        for i in range(num_output):
            output_results[i] = collect_results_cpu(results[i], len(dataset), tmpdir)
        #results[1] = collect_results_cpu(results[1], len(dataset), tmpdir)
    return output_results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist(f'.dist_test')
            tmpdir = tempfile.mkdtemp(dir=f'.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loding pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir) 
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None

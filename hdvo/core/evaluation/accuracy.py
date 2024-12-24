import numpy as np


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

# liuziming 21/3/3
def rmse(scores, labels, frame_dir, folder_list, items=('translation', 'rotation')):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = dict()

    labels = np.array(labels)#[:, np.newaxis]
    scores = np.array(scores)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for i in folder_list:
        for j in range(scores.shape[0]):
            if int(i) == int(frame_dir[j]):
                if i not in groups.keys():
                    groups[i] = [[],[]]
                groups[i][0].append(labels[j]) # save gt pose
                groups[i][1].append(scores[j]) # save pred pose
    #print(groups['04'])
    #print(groups['04'][0])
    res['overall_rotation'] = 0 
    res['overall_traslation'] = 0 
    for i in groups.keys():
        seqi_labels = np.array(groups[i][0])
        seqi_scores = np.array(groups[i][1])
        if 'rotation' in items:
            mse_rotate = 100 * np.mean((seqi_scores[:, :3] - seqi_labels[:, :3])**2)
            res[i+'_rotation'] = mse_rotate
            res['overall_rotation'] +=mse_rotate
        if 'translation' in items:
            mse_translate = np.mean((seqi_scores[:, 3:] - seqi_labels[:, 3:6])**2)
            res[i+'_translation'] = mse_translate
            res['overall_traslation'] += mse_translate
    res['overall_rotation'] /= len(folder_list)
    res['overall_traslation'] /= len(folder_list)
    
    #out = [res['overall_rotation'], res['overall_traslation']]
    
    #for i in folder_list:
    #    if 'rotation' in items:
    #        out.append(res[i+'_ratation'])
    #    if 'translation' in items:
    #        out.append(res[i+'_translation'])
    return res


# liuziming 21/06/06
def rel(pred_depth, gt_depth, frame_dir, folder_list, ):
    """Calculate depth relative error score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs

    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys()),"{} and {}".format(folder_list,groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        rel_error = np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)/gt_depth_i, axis=1), axis=0)
        res['{}'.format(i)] = rel_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    res['dataset'] = np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)/gt_depth_i, axis=1), axis=0)
    

    return res


# liuziming 21/06/06
def relsqr(pred_depth, gt_depth, frame_dir, folder_list, ):
    """Calculate depth relative square error score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs

    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        rel_sqr_error = np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)**2/gt_depth_i, axis=1), axis=0)
        res['{}'.format(i)] = rel_sqr_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    res['dataset'] = np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)**2/gt_depth_i, axis=1), axis=0)
    
    return res

# liuziming 21/06/06
def log10(pred_depth, gt_depth, frame_dir, folder_list, ):
    """Calculate depth log 10 error score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs

    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        rel_sqr_error = np.mean( np.mean(np.abs(np.log10(pred_depth_i)-np.log10(gt_depth_i)), axis=1), axis=0)
        res['{}'.format(i)] = rel_sqr_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    res['dataset'] = np.mean( np.mean(np.abs(np.log10(pred_depth_i)-np.log10(gt_depth_i)), axis=1), axis=0)
    
    return res

# liuziming 21/06/06
def rmsedepth(pred_depth, gt_depth, frame_dir, folder_list, ):
    """Calculate depth root mean square  error score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs

    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        rmse_error = np.sqrt(np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)**2, axis=1), axis=0))
        res['{}'.format(i)] = rmse_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    res['dataset'] = np.sqrt(np.mean( np.mean(np.abs(pred_depth_i-gt_depth_i)**2, axis=1), axis=0))
    
    return res

# liuziming 21/06/06
def rmsedepthlog(pred_depth, gt_depth, frame_dir, folder_list, ):
    """Calculate depth logarithmic root mean square  error score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs

    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        rmselog_error = np.sqrt(np.mean( np.mean(np.abs(np.log10(pred_depth_i)-np.log10(gt_depth_i))**2, axis=1), axis=0))
        res['{}'.format(i)] = rmselog_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    res['dataset'] = np.sqrt(np.mean( np.mean(np.abs(np.log10(pred_depth_i)-np.log10(gt_depth_i))**2, axis=1), axis=0))
    
    return res

# liuziming 21/06/06
def correct(pred_depth, gt_depth, frame_dir, folder_list, tau=[1.25,1.25**2,1.25**3]):
    """Calculate depth correct score.

    Args:
        pred_depth (list[np.ndarray]): 
        gt_depth (list[int]): Ground truth depth
        frame_dir (tuple[int]): seuqence director
        folder_list (tuple[int]): all sequences dirs
        tau int: 1.25 or 1.25**2 or 1.25**3
    Returns:
        dict[float]: {'dataset': , 'seq0': , 'seq1': ...}
    """

    pred_depth = np.array(pred_depth)#[:, np.newaxis]
    gt_depth = np.array(gt_depth)

    frame_dir = list(frame_dir)
    # group them
    groups = {} #{f:[[],[]] for f in folder_list} 
    for seqi in folder_list: # sequence name 
        for n in range(pred_depth.shape[0]):
            if seqi == frame_dir[n]:
                if seqi not in groups.keys(): # initial a new sequence
                    groups[seqi] = [[],[]]
                groups[seqi][0].append(gt_depth[n]) # save gt depth
                groups[seqi][1].append(pred_depth[n]) # save pred depth
    #print(groups['04'])
    #print(groups['04'][0])
    res = dict()
    dataset_gt = []
    dataset_pred = []
    #assert len(folder_list) == len(groups.keys())
    for i in groups.keys(): # for each sequence
        gt_depth_i = np.array(groups[i][0])
        Ni = gt_depth_i.shape[0]
        gt_depth_i = gt_depth_i.reshape((Ni,-1))
        dataset_gt.append(gt_depth_i)
        pred_depth_i = np.array(groups[i][1])
        pred_depth_i = pred_depth_i.reshape((Ni,-1))
        dataset_pred.append(pred_depth_i)
        for tau_val in tau:
            correct_error = np.mean( np.mean(np.maximum(pred_depth_i/gt_depth_i, gt_depth_i/pred_depth_i)<tau_val, axis=1), axis=0)
            res['{}_{}'.format(i,tau_val)] = correct_error

    dataset_gt = np.concatenate(dataset_gt,axis=0)
    dataset_pred = np.concatenate(dataset_pred, 0)
    for tau_val in tau:
        res['dataset_{}'.format(tau_val)] = np.mean( np.mean(np.maximum(pred_depth_i/gt_depth_i, gt_depth_i/pred_depth_i)<tau_val, axis=1), axis=0)
    
    return res

 
def epe(disp_est, disp_gt, eval_range):
    epe_ = []
    for i in range(len(disp_est)):
        mask = (disp_gt[i]>eval_range[0]) & (disp_gt[i]<eval_range[1])
        if len(disp_est[i][mask])==0:
            epe_.append(0)
            print("no value after masking")
            continue
        epe_.append(np.mean(np.abs(disp_est[i][mask] - disp_gt[i][mask])))
    return np.mean(epe_)

def epe2(disp_est, disp_gt, eval_range):
    epe_ = []
    for i in range(len(disp_est)):
        epe = np.abs(disp_est[i] - disp_gt[i])
        epe = epe.flatten()
        val = (disp_gt[i].flatten() >= 0.5) & (disp_gt[i].flatten() <= 192)
        if(np.isnan(epe[val].mean().item())):
            continue
        epe_.append(epe[val].mean().item())

    return np.mean(np.array(epe_))
 

def n_disps_epe(disp_est, disp_gt, eval_range):
    
    N_epes = []
    for lb in range(eval_range[0], eval_range[1], 6):
        hb = lb + 6
        epe_ = []
        for i in range(len(disp_est)):
            mask = (disp_gt[i]>=lb) & (disp_gt[i]<hb)
            if len(disp_est[i][mask])==0:
                #epe_.append(0)
                #print("no value after masking")
                continue
            epe_.append(np.mean(np.abs(disp_est[i][mask] - disp_gt[i][mask])))
        if len(epe_)==0:
            N_epes.append(0)
            continue
        N_epes.append(np.mean(epe_))
    return N_epes

def count_disp_dist(disp_est, disp_gt, eval_range):
    
    N_pixels = []
    for lb in range(eval_range[0], eval_range[1], 6):
        hb = lb + 6
        epe_ = []
        for i in range(len(disp_est)):
            mask = (disp_gt[i]>=lb) & (disp_gt[i]<hb)
            if len(disp_est[i][mask])==0:
                #epe_.append(0)
                #print("no value after masking")
                continue
            epe_.append(np.sum(mask))
        if len(epe_)==0:
            N_pixels.append(0)
            continue
        N_pixels.append(np.sum(epe_))
    return N_pixels

def d1(disp_est, disp_gt, eval_range):
    d1_ = []
    for i in range(len(disp_est)):
        mask = (disp_gt[i]>eval_range[0]) & (disp_gt[i]<eval_range[1])
        D_est = disp_est[i][mask]
        D_gt = disp_gt[i][mask]
        if len(disp_est[i][mask])==0:
            d1_.append(0)
            print("no value after masking")
            continue
        E = np.abs(D_est - D_gt)
        err_mask = (E > 3) & (E / np.abs(D_gt) > 0.05)
        d1_.append(np.mean(err_mask.astype(float)))
    return np.mean(d1_)


def three_pe(disp_est, disp_gt, eval_range):
    three_pe_ = []
    for i in range(len(disp_est)):
        mask = (disp_gt[i]>eval_range[0]) & (disp_gt[i]<eval_range[1])
        D_est = disp_est[i][mask]
        D_gt = disp_gt[i][mask]
        if len(disp_est[i][mask])==0:
            three_pe_.append(0)
            print("no value after masking")
            continue
        E = np.abs(D_est - D_gt)
        err_mask = (E > 3)  
        three_pe_.append(np.mean(err_mask.astype(float)))
    return np.mean(three_pe_)


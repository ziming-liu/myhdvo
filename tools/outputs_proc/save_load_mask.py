'''
Author: Ziming Liu
Date: 2022-06-16 11:35:53
LastEditors: Ziming
LastEditTime: 2022-09-23 12:59:14
Description: ...
Dependent packages: don't need any extral dependency
'''
import os
import cv2
import numpy as np




def save_mask_maps(dataset_type, mask_type, test_seq_id, pred_mask, work_dir, epoch, first_frame_id=0, iftransparent=False ):
    '''
    description: 
    parameters: dataset_type: "KittiDepthStereoDataset""KittiStereoMatchingDataset" are static images dataset. we give a pseudo seqID 99.
                mask_type: "mask_occlusion_stereo", "mask_occlusion_temporal", 

    return: {*}
    '''    
    num_samples = len(pred_mask)
    if dataset_type == "KittiDepthStereoDataset" or dataset_type == "KittiStereoMatchingDataset":
        test_seq_id = "99"
    else:
        test_seq_id = test_seq_id

    mask_dir = os.path.join(work_dir,epoch+f"_pred_{mask_type}"+ test_seq_id)
    if os.path.exists(mask_dir):
        import shutil
        shutil.rmtree(mask_dir)
        os.makedirs(mask_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    print("saving {} pred depths into {}".format(num_samples, mask_dir))

     
    for i in range(num_samples):
        pred_mask[i] = pred_mask[i]*255
        #pred_mask[i] = pred_mask[i].astype(np.uint16) # meter to centen meter
        #print(pred_mask[i])
        # save pred depth into work_dirs 
        save_map_path = os.path.join(mask_dir, '{:0>6}.png'.format(first_frame_id+i))
        if len(pred_mask[i].shape)==3:
            pred_mask[i] = np.squeeze(pred_mask[i], 0) 
        elif len(pred_mask[i].shape)==2:
            pass
        else:
            raise ValueError
        img = pred_mask[i].squeeze()
        if iftransparent:
            h,w = img.shape
            img = img.reshape((h,w,1))
            img = img.repeat(3, axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
            img[img==1] =255
            img_a = img[:,:,3:]
            img_bgr = img[:,:,:3].mean(2,keepdims=True)
            #print(img_bgr==255)
            img[:,:,3:][img_bgr==255] = 180
            #print(img[:,:,3:])
            #img = np.concatenate((img[:,:,:3],img_a),2)
        cv2.imwrite(save_map_path, img ) # save depth into png 
        
    

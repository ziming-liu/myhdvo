'''
Author: Ziming Liu
Date: 2022-06-17 15:44:51
LastEditors: Ziming
LastEditTime: 2022-06-30 14:30:53
Description: ...
Dependent packages: don't need any extral dependency
'''
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import os


def vis_error(image_a, image_b, error_value, error_map, save_name, save_path):
    '''
    description: image_a, image_b, tensor image, bxcxhxw or cxhxw, image should be ranged in [0,255]
                error_value, scalar
                error_map, bxcxhxw or cxhxw
    return: {*}
    ''' 
    if len(image_a.shape)==4:
        image_a = image_a[0]
        image_b = image_b[0]
        error_map = error_map[0]
    assert len(image_a.shape)==3 and len(image_b.shape)==3 and len(error_map.shape)==3
    # to show image rgb, permute dimensions
    image_a, image_b = image_a.detach().cpu().numpy(), image_b.detach().cpu().numpy()
    image_a = np.transpose(image_a, (1,2,0))
    image_b = np.transpose(image_b, (1,2,0))
    fig = plt.figure(figsize=(16,2))
    ax = plt.subplot(1,3,1)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(image_a.astype(np.uint8), cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('x')

    ax = plt.subplot(1,3,2)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(image_b.astype(np.uint8), cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('y')

    ax = plt.subplot(1,3,3)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(error_map.detach().cpu().numpy()[0], cmap='jet_r')
    fig.colorbar(im, cax=cax)
    im.set_clim(-1e-5, 1e-5)
    ax.set_title('NCC(x,y): %.4f' % error_value.detach().cpu().numpy())
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_normalized_cross_correlation_2d_%s.png' % save_name))
    plt.close()
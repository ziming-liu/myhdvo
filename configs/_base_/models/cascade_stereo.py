import os.path as osp

max_disp = 192
model=dict( 
        type="CascadeStereoPSMNet",
        maxdisp=max_disp,
        ndisps=[48,24],
        disp_interval_pixel=[4,1], 
        using_ns=True,  # using neighbor search
        ns_size=3, 
        grad_method="detach",
        cr_base_chs=[32, 32, 16], # cost regularization base channels
)
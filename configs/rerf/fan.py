_base_ = '../default.py'

expname = 'fan'
basedir = './output'

train_mode = 'sequential'
fix_rgbnet = True
frame_num = 97

res_lambda=1e-2
deform_lambda=1e-2

# added by jhpark
deform_res_mode= 'separate'
use_res = False
use_deform = ''

data = dict(
    datadir= './data/fan',                 # path to dataset root folder
    dataset_type= 'llff',        # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=True,     # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=False,               # [TODO]
    # Below are forward-facing llff specific settings. Not support yet.
    ndc=True,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    factor=1,                     # [TODO]
    llffhold=0,                   # testsplit
    load_depths=False,            # load depth
    
    movie_render_kwargs={
        'scale_r': 0.8, # circling radius
        'scale_f': 8.0, # the distance to the looking point of foucs
        'zdelta': 0.1,  # amplitude of forward motion
        'zrate': 0.1,   # frequency of forward motion
        'N_rots': 1.0,    # number of rotation in 120 frames
        'N_views' : 97 # frame nums
    }
)

deform_from_start=False

coarse_train = dict(
    N_iters=0, 
)


fine_train=dict(
    N_iters=30000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    pg_scale_pretrained=[ 2000,4000,6000,8000],
    decay_after_scale=0.1,
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
    lrate_deformation_field=1e-4
    
)




_mpi_depth = 216
_stepsize = 1.0

fine_model_and_render = dict(
    maskout_near_cam_vox=False,
    num_voxels=384*384*_mpi_depth,
    mpi_depth=_mpi_depth,
    stepsize=_stepsize,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=_stepsize/_mpi_depth/5,
)
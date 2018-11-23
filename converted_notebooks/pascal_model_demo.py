import sys
sys.path.insert(0, '../')
import os
os.listdir('../data/pascal3d_mixture_models')
from datasets import pascal3d
from models.infinite_mixture import BiternionMixture
import numpy as np
from utils import angles
import matplotlib.pyplot as plt
#matplotlib inline

cls = 'aeroplane' # if cls is None, all classes will be loaded
pascaldb_path = '../data/pascal3d+_imagenet_train_test.h5'
x_train, y_train, x_val, y_val, x_test, y_test = pascal3d.load_pascal_data(pascaldb_path, cls=cls)

# if you want to use pretrained Inception\Resnet models as a backbone and experience problems
# with Keras automatically downloading the weights (SSL certiticate issues), download it manually from here:
#
# https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
# https://github.com/keras-team/keras-applications/releases/download/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5
#
# and place it to $HOME/.keras/models



model = BiternionMixture(z_size=8, backbone_cnn='inception', hlayer_size=512, n_samples=50)
model.model.load_weights('../data/pascal3d_mixture_models/bimixture_aeroplane_22dd6663baaa4b036f73_hls512_zs_8_ns_50.1e.h5')

az_deg, el_deg, tilt_deg = model.predict(x_test[0:10])
xvals, az_pdf = model.pdf(x_test[0:10], gamma=1.0e-1, angle='azimuth')



fid = 9
plt.axvline(angles.bit2rad(y_test)[fid], c='orange', label='ground truth')
plt.axvline(np.deg2rad(az_deg[fid]), c='darkblue', label='prediction')
plt.plot(xvals, az_pdf[fid], label='predicted density')
plt.legend()

fig_path = '../../logs/detection_examples'

if not os.path.exists(fig_path):
    os.mkdir(fig_path)

model.visualize_detections_on_circle(x_test[0:10], y_true=y_test, save_figs=True, save_path=fig_path)

save_dir = '../../logs/vp_test_results/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model.save_detections_for_official_eval(x_test, os.path.join(save_dir, '%s_pred_view.txt'%cls))
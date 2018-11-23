import sys
sys.path.insert(0, '../')
from datasets import pascal3d
from models.infinite_mixture import BiternionMixture
import os

pascaldb_path = '../data/pascal3d+_imagenet_train_test.h5'
cls = 'car' # if cls is None, all classes will be loaded

x_train, y_train, x_val, y_val, x_test, y_test = pascal3d.load_pascal_data(pascaldb_path, cls=cls)

model = BiternionMixture(z_size=2, backbone_cnn='inception', hlayer_size=512, n_samples=50)

ckpt_path = '../logs/$s.h5'

model.fit(x_train, y_train, validation_data=[x_val, y_val], ckpt_path=ckpt_path, epochs=50, patience=5)

model.model.load_weights(ckpt_path)

model.evaluate(x_test, y_test)

save_dir = '../logs/vp_test_results/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model.save_detections_for_official_eval(x_test, os.path.join(save_dir, '%s_pred_view.txt' % cls))

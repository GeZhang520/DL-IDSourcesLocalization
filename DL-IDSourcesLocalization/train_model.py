import os
import numpy as np
import matplotlib.pyplot as plt
from model import IDSourceEstimator
import hdf5storage as hdf5

Estimator = IDSourceEstimator(input_shape=(10, 10, 2),
                              angle_units=2,
                              spread_units=2)
Estimator.plotModelStructure(file_path='./outputs/model.jpg')

# ---------- load dataset ---------------
# mat_path = os.getcwd() + '\\matlab\\dataset\\'
# data_mat = hdf5.loadmat(mat_path + 'train.mat')
# train_data = data_mat['signal_cov']
# train_angle_labels = data_mat['angle_labels']
# train_spread_labels = data_mat['spread_labels']
# # ---------------------------------------
#
# # ------------- train -------------------
# history = Estimator.fit(
#     train_inputs=train_data,
#     train_angle_labels=train_angle_labels,
#     train_spread_labels=train_spread_labels
# )
# # ---------------------------------------
#
# # save model
# Estimator.saveWeights(file_path='./outputs/model.h5')
# # --------- plot epoch-loss -------------
# epochs = np.array(history.epoch)[1:len(history.epoch)]
# plt.figure(1)
# plt.plot(epochs, np.array(history.history['loss'])[epochs], color='y', label='train loss')
# plt.plot(epochs, np.array(history.history['val_loss'])[epochs], color='c', label='val loss')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('loss')
# plt.grid()
# plt.show()
# plt.savefig('./outputs/count_loss.png')
# ---------------------------------------

# ---------- test outputs ---------------
mat_path = os.getcwd() + '\\matlab\\dataset\\'
data_mat = hdf5.loadmat(mat_path + 'test.mat')
test_data = data_mat['signal_cov_test']
test_angle_labels = data_mat['angle_labels_test']
test_spread_labels = data_mat['spread_labels_test']
signal_num = int(data_mat['signal_num'][0][0])

# load model weights
Estimator.loadWeights(file_path='./outputs/model.h5')
# predict on batch
angles, spread = Estimator.predictOnBatch(inputs=test_data, signal_num=signal_num)

# ----------- print results ------------
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print('nominal DOA labels:\n', test_angle_labels)
print('nominal DOA pred:\n', angles, '\n')
print('angular spread labels:\n', test_spread_labels)
print('angular spread pred:\n', spread)
# ---------------------------------------

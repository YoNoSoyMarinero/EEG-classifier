import numpy as np
import scipy.io as sio

class RawData:
    #Loading data and formating all singals into numpy arrays
    def __init__(self, train_name: str, test_name: str) -> None:
        emg_train_mat = sio.loadmat(train_name)
        emg_test_mat = sio.loadmat(test_name)
        self.emg_train_chs: np.ndarray = np.array(emg_train_mat['CHS_vezbe'])
        self.emg_train_type: np.ndarray = np.array(emg_train_mat['grasp_type_vezbe'])
        self.emg_train_t: np.ndarray = np.array(emg_train_mat['t_vezbe'])

        self.emg_test_chs: np.ndarray = np.array(emg_test_mat['CHS_provera'])
        self.emg_test_type: np.ndarray = np.array(emg_test_mat['grasp_type_provera'])
        self.emg_test_t: np.ndarray = np.array(emg_test_mat['t_provera'])
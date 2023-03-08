import numpy as np

class FeatureExtraction:

    def calculate_mav(self, sub_array) -> float:
        return np.mean(np.abs(sub_array))
    
    def calculate_rms(self, sub_array) -> float:
        return np.sqrt(np.mean(np.square(sub_array)))
    
    def calculate_wl(self, sub_array) -> float:
        return np.sum(np.abs(np.diff(sub_array)))

    @classmethod
    def featutre_extraction(self, emg_chs: np.ndarray, emg_type: np.ndarray, freq_sample:int= 2000, kernel_length:int = 250) -> np.ndarray:
        step = int(freq_sample*kernel_length/1000)
        data_set: np.ndarray = np.empty((0, 25), float)
        temp_vector = []
        for i in range(0, emg_chs.shape[1], int(step/2)):
            for j in range(emg_chs.shape[0]):
                temp_vector.append(self.calculate_mav(self, emg_chs[j, i:i+step - 1]))
                temp_vector.append(self.calculate_rms(self, emg_chs[j, i:i+step - 1]))
                temp_vector.append(self.calculate_wl(self, emg_chs[j, i:i+step - 1]))

            if (emg_type[0, i:i+step-1][0] != emg_type[0, i:i+step-1][-1]):
                temp_vector = []
                continue

            temp_vector.append(np.median(emg_type[0,i:i+step-1]))
            vector = np.array(temp_vector).reshape(1,25)
            data_set = np.append(data_set, vector, axis=0)
            temp_vector = []

        return data_set
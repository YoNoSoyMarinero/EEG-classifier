import pandas as pd

class EMGDataFrame:

    def column_label_generator(self) -> list:
        features = ["mav", "rms", "wl"]
        column_labels = []
        for i in range(8):
            for feature in features:
                column_labels.append(feature + str(i + 1))

        column_labels.append("type")
        return column_labels

    def __init__(self, train_data_np, test_data_np) -> None:
        self.train_df = pd.DataFrame(train_data_np, columns=self.column_label_generator())
        self.test_df = pd.DataFrame(test_data_np, columns=self.column_label_generator())
import pandas as pd

class DataReader:
    def __init__(self, file_path):
        try:
            self.__df = pd.read_pickle(file_path)
            self.__capacity = self.__df.index[-1][0] + 1
        except FileNotFoundError:
            print(f'File {file_path} not found.')

    def header(self):
        return self.__df.to_numpy().ravel()[0]

    def data(self):
        return self.__df.to_numpy().ravel()[1:]

    def __len__(self):
        return self.__capacity


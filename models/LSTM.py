import tensorflow as tf
from time_series_validation import *
import logging
from  Models import *
import matplotlib as mpl
import matplotlib.pyplot as plt

class TimeSeries():
    def __init__(self):
        self.inputFile = 'covid_data_copy.csv'
        self.fileFormat = "csv"
        self.inputColumns = ['date', 'state', 'new_case', 'inpatient_beds_used', 'inpatient_beds_used_covid', 'tot_cases', 'Administered']
        self.DataFrame = None
        self.Model = None

    def readFile(self,fileName, fileFormat = "csv"):
        self.inputFile = fileName
        if fileFormat == "csv":
            try:
                self.DataFrame=pd.read_csv(self.inputFile, usecols=self.inputColumns)
                self.column_indices  = {name: i for i, name in enumerate(self.DataFrame.columns)}
            except:
                logging.error("Failed to read file:{}".format(err))

    def setColumns(self,columnList):
        self.inputColumns = columnList

    def getColumnIndices(self):
        return self.column_indices

    def setDataFrame(self, dataFrame):
        self.DataFrame = dataFrame
        self.date_time = self.DataFrame.pop("date")
        self.DataFrame = self.DataFrame.drop(columns="day_number")
        self.column_indices  = {name: i for i, name in enumerate(self.DataFrame.columns)}

    def displayColumns(self, columns, **kwargs):
        """ Display the variables over time period"""
        for column in columns:
            if column not in self.DataFrame.columns:
                logging.warning('"{}" not found in dataframe')
        
        plot_cols = columns
        plot_features = self.DataFrame[plot_cols]
        plot_features.index = self.date_time
        _ = plot_features.plot(subplots=True)

    def splitData(self, fraction = [0.7,0.2,0.1]):
        if len(fraction) != 3:
            logging.warning("Three parameters expected for training/validation/test perspectively ,using default setting")
            fraction = [0.7,0.2,0.1]

        n = len(self.DataFrame)
        train_fraction = float(sum(fraction[:1]))/sum(fraction)
        val_fraction = float(sum(fraction[:2]))/sum(fraction)
        test_fraction = float(sum(fraction[:3]))/sum(fraction)
        self.train_df = self.DataFrame[0:int(n*train_fraction)]
        self.val_df = self.DataFrame[int(n*train_fraction):int(n*(val_fraction))]
        self.test_df = self.DataFrame[int(n*(val_fraction)):]
        self.num_features = self.DataFrame.shape[1]

    def normalizeData(self):
        train_mean = self.train_df.mean()
        train_std = self.train_df.std()

        self.train_df = (self.train_df - train_mean) / train_std
        self.val_df = (self.val_df - train_mean) / train_std
        self.test_df = (self.test_df - train_mean) / train_std

    def setWindowLength(self, input_width = 14, label_width = 1, shift = 1, label_columns = ["inpatient_beds_used"]):
        self.label_columns = label_columns
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(self.train_df.columns)}
                                    #enumerate(label_columns)}
        self.inputColumns_indices = {name: i for i, name in
                           enumerate(self.train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift


    def generateWindow(self):
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.label_columns_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def createWindow(self):
        self.window = tf.stack([np.array(self.train_df[:self.total_window_size]),
                                np.array(self.train_df[100:100+self.total_window_size]),
                                np.array(self.train_df[200:200+self.total_window_size])])
        inputs, labels = self.split_window(self.window)
        print(inputs,labels)

    def setDataset(self, batch_size = 16, sequence_stride=1, shuffle = True):
        self.batch_size = batch_size
        self.sequence_stride = sequence_stride
        self.shuffle = shuffle

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=self.shuffle,
            batch_size=self.batch_size)

        ds = ds.map(self.split_window)
        return ds


    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self,'_example',None)
        result = None
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result


    def setModel(self):
        pass

    def setOptimizer(self):
        pass

    def plot(self, model=None, plot_col='beds', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, -(len(self.label_indices)):],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, -(len(self.label_indices)):],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    
if __name__ == "__main__":
    covid_state, state_list = load_states()
    covid_state["AZ"].head()
    ts = TimeSeries()
    ts.setDataFrame(covid_state["AZ"][1:])
    ts.setColumns(["beds","cases","vaccines","cases_7day"])
    ts.splitData()
    ts.normalizeData()
    ts.setWindowLength(label_columns=["beds"])
    ts.setDataset(batch_size=32)
    ts.setWindowLength(input_width = 14, label_width = 1, shift = 2)
    ts.generateWindow()
    ts.createWindow()   

    



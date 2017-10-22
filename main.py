import argparse
import sys
from random import randint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from data import load_data, prepared_data
import h5py

# Keras f1 code taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.

#         Only computes a batch-wise average of recall.

#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         """Precision metric.

#         Only computes a batch-wise average of precision.

#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision

#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall))

class Options(object):
    """Class to represent command options"""
    def __init__(self, typ, epochs, resample, dropout, layers, num_per_class, weight_data, boost_per_class, seed, min_loss):
        self.typ = typ
        self.epochs = epochs
        self.resample = resample
        self.dropout = dropout
        self.layers = layers
        self.num_per_class = num_per_class
        self.weight_data = weight_data
        self.boost_per_class = boost_per_class
        self.seed = seed
        self.min_loss = min_loss

    def _describe_nums(self):
        if self.num_per_class is None:
            return 'NA'
        return '.'.join(map(lambda x: '{}-{}'.format(x[0], x[1]), self.num_per_class.items()))

    def describe_boost_per_class(self):
        if not self.weight_data:
            return 'NA'
        return '.'.join(map(lambda x: '{}-{}'.format(x[0], x[1]), self.boost_per_class.items()))

    def describe(self):
        """return a string description of the options to use for report and model names"""
        return 'T{}-E{}-R{}-D{}-L{}-N{}-W{}-B{}-S{}-M{}'.format(
            self.typ,
            self.epochs,
            self.resample,
            self.dropout,
            ".".join(map(lambda x: str(x), self.layers)),
            self._describe_nums(),
            self.weight_data,
            self.describe_boost_per_class(),
            self.seed,
            self.min_loss
        )

class OptionBuilder():
    """Builder object for Options class"""
    #pylint: disable=C0111
    def __init__(self):
        self._typ = None
        self._epochs = None
        self._resample = None
        self._dropout = None
        self._layers = None
        self._weight_data = None
        self._boost_per_class = None
        self._seed = None
        self._min_loss = None

    def typ(self, typ):
        self._typ = typ
        return self

    def epochs(self, epochs):
        self._epochs = epochs
        return self

    def resample(self, resample):
        self._resample = resample
        return self

    def dropout(self, dropout):
        self._dropout = dropout
        return self

    def layers(self, layers):
        self._layers = layers
        return self

    def num_per_class(self, num_per_class):
        self._num_per_class = num_per_class
        return self

    def weight_data(self, weight_data):
        self._weight_data = weight_data
        return self

    def boost_per_class(self, boost_per_class):
        self._boost_per_class = boost_per_class
        return self

    def seed(self, seed):
        self._seed = seed
        return self

    def min_loss(self, min_loss):
        self._min_loss = min_loss
        return self

    def options(self):

        if self._weight_data and self._resample is not None:
            raise ValueError("cannot use weights and resampling at the same time")

        if self._typ == 'modern' and self._layers is None:
            raise ValueError("layers option required for modern type")

        num_per_class = None
        if self._num_per_class:
            if len(self._num_per_class) != 7:
                raise ValueError("if num per class is provided it must have 7 entries")

            num_per_class = {i + 1: v for i, v in enumerate(self._num_per_class)}

        boost_per_class = {n: 1 for n in range(7)}
        if self._boost_per_class and not self._weight_data:
            raise ValueError("boost per class only works with weighting")

        if self._boost_per_class:
            if len(self._boost_per_class)  != 7:
                raise ValueError("if boost per class is provided it must have 7 entries")

            boost_per_class = {i: v for i, v in enumerate(self._boost_per_class)}

        return Options(self._typ,
                       self._epochs,
                       self._resample,
                       self._dropout,
                       self._layers,
                       num_per_class,
                       self._weight_data,
                       boost_per_class,
                       self._seed,
                       self._min_loss)

class Data(object):
    """A class to handle the preparation of train/test data"""
    def __init__(self, options):
        self.options = options
        self.data = load_data()
        self.features, self.labels = prepared_data(self.data)
        self.x_train = None
        self.y_train = None
        self.y_train_ary = None
        self.x_test = None
        self.y_test = None
        self.y_test_ary = None
        self.resampler = None

    def create_train_test_data(self):
        """Create the train test split data from features and labels"""
        self.x_train, self.x_test, self.y_train_ary, self.y_test_ary = train_test_split(
            self.features,
            self.labels,
            test_size=0.2,
            random_state=options.seed,
            stratify=self.labels)

        self._postprocess_data()

        self.y_train_ary -= 1
        self.y_test_ary -= 1

        self.y_train = keras.utils.to_categorical(self.y_train_ary, num_classes=7)
        self.y_test = keras.utils.to_categorical(self.y_test_ary, num_classes=7)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def compute_class_weight(self):
        """Returns the class weights for the TRAINING data.
        If the class weight option is true then returns the class weights calculated by
        the 'balanced' option. Otherwise all should be 1"""
        typ = 'balanced' if self.options.weight_data else None
        cw = {k: options.boost_per_class[k] * v for k, v in enumerate(compute_class_weight(typ, np.unique(self.y_train_ary), self.y_train_ary))}
        print("Using class weights:", cw)
        return cw

    def compute_sample_weight(self):
        """Returns the sample weights for the TESTING data"""
        sw = compute_sample_weight('balanced', self.y_test_ary)
        print("Using sample weights (only first 10 shown):", sw[:10])
        return sw

    def filter_by_category(self, x, y, category):
        """Return features and labels for a single category"""
        features_and_labels = np.hstack((x, y))
        filtered = np.array([r for r in features_and_labels if r[53 + category] == 1])
        features, labels = filtered[...,:54], filtered[...,54:]
        return features, labels

    def _postprocess_data(self):
        if self.resampler is not None:
            print("Postprocessing")
            self.x_train, self.y_train_ary = self.resampler.fit_sample(self.x_train, self.y_train_ary)

class RandomUnderSampledData(Data):
    """A class to handle the preparation of Random Undersampled train/test data"""
    def __init__(self, options):
        Data.__init__(self, options)
        ratio = options.num_per_class if options.num_per_class is not None else 'auto'
        self.resampler = RandomUnderSampler(ratio=ratio, random_state=self.options.seed)
        print("Using random under sampler")

class RandomOverSampledData(Data):
    """A class to handle the preparation of Random Oversampled train/test data"""
    def __init__(self, options):
        Data.__init__(self, options)
        self.resampler = RandomOverSampler(random_state=self.options.seed)
        print("Using random over sampler")

def create_data_handler(options):
    """factory function to create appropriate data hander based upon options"""
    if options.resample == 'undersample':
        return RandomUnderSampledData(options)
    elif options.resample == 'oversample':
        return RandomOverSampledData(options)
    else:
        return Data(options)

class ModelMaker(object):
    def __init__(self, options, x_shape):
        self._x_shape = x_shape
        self._options = options

    def create_model():
        raise NotImplementedError()

class OriginalModelMaker(ModelMaker):

    def create_model(self):
        model = Sequential()
        model.add(Dense(120, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=1.0), input_dim=self._x_shape[1]))
        model.add(Dense(7, activation='sigmoid'))

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.5)
        model.compile(loss='mean_squared_error',
            optimizer=sgd,
            metrics=['accuracy'],
            weighted_metrics=['accuracy'])

        return model

class ModernModelMaker(ModelMaker):

    def _add_layer(self, n_units, input_dim=None):
        """Add a dense layer + a dropout layer to the model"""

        if input_dim is not None:
            self._model.add(Dense(n_units, activation='relu', input_dim=input_dim))
        else:
            self._model.add(Dense(n_units, activation='relu'))

        self._model.add(Dropout(self._options.dropout))

    def create_model(self):
        """Create a model using modern parameters, with an architecture dictated by pattern.
        Pattern should be an array of (dimensions, dropout_rate) tuples, one for each hidden layer"""

        self._model = Sequential()
        is_first_layer = True
        for num_units in self._options.layers:
            if is_first_layer:
                self._add_layer(num_units, self._x_shape[1])
                is_first_layer=False
            else:
                self._add_layer(num_units)

        self._model.add(Dense(7, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self._model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy'])
        return self._model

def create_model_maker(options, x_shape):
    if options.typ == 'modern':
        return ModernModelMaker(options, x_shape)

    return OriginalModelMaker(options, x_shape)

class Trainer(object):
    """Base class for training and testing a model"""
    def __init__(self, options, data, model):
        self._options = options
        self._data = data
        self._model = model
        self._epoch_logger = None

    def train(self):
        """Train the model using the training data"""
        self.epoch_logger = EpochEndCallback()
        early_stopping = EarlyStoppingWithMinLoss(self._options.min_loss)
        self._data.create_train_test_data()
        self._model.fit(self._data.x_train, self._data.y_train,
            epochs=self._options.epochs,
            batch_size=128,
            verbose=0,
            class_weight=self._data.compute_class_weight(),
            callbacks=[self.epoch_logger, early_stopping])



class TestResults(object):
    """container for results of testing a model"""
    def __init__(self):
        self.loss = 0.
        self.accuracy = 0.
        self.weighted_accuracy = 0.
        self.categorical_results = {}

    def add_results_for_category(self, category, accuracy):
        self.categorical_results[category] = accuracy

class Tester(object):
    def __init__(self, data, model):
        self._data = data
        self._model = model

    def _test_with(self, x, y, sample_weight=None):
        return self._model.evaluate(
            x,
            y,
            batch_size=128,
            verbose=0,
            sample_weight=sample_weight)

    def _test_complete(self):
        loss, acc, weighted_acc = self._test_with(
            self._data.x_test,
            self._data.y_test,
            sample_weight=self._data.compute_sample_weight())
        self.results.loss = loss
        self.results.accuracy = acc
        self.results.weighted_accuracy = weighted_acc

    def _test_by_category(self):
        for c in range(1, 8):
            _, acc, _ = self._evaluate_by_category(c)
            self.results.add_results_for_category(c, acc)

    def _evaluate_by_category(self, category):
        """Perform an evaluation on the model for a single category"""
        filtered_features, filtered_labels = self._data.filter_by_category(self._data.x_test, self._data.y_test, category)
        return self._test_with(filtered_features, filtered_labels)

    def test(self):
        self.results = TestResults()
        self._test_complete()
        self._test_by_category()
        return self.results

class EpochEndCallback(Callback):
    """Logs epoch number, loss and accuracy at the end of each epoch"""
    def __init__(self):
        self.loss = []
        self.acc = []
        self.weighted_acc = []


    def on_epoch_end(self, epoch, logs=None):
        """Print out logs at the end of each epoch"""
        if logs is None:
            print("No logs provided")
            return

        print("Epoch", epoch)
        print("Loss", logs.get('loss'))
        print("Accuracy", logs.get('acc'))
        print("Weighted Accuracy", logs.get('weighted_acc'))
        print("==============================")

        # record the values.
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.weighted_acc.append(logs.get('wieghted_acc'))

    def save(self, filename):
        f = h5py.File(filename, 'w')
        loss = np.array(self.loss).astype('float32')
        acc = np.array(self.acc).astype('float32')
        weighted_acc = np.array(self.weighted_acc).astype('float32')
        f.create_dataset('loss', data=loss)
        f.create_dataset('acc', data=acc)
        f.create_dataset('weighted_acc', data=weighted_acc)
        f.close()

class EarlyStoppingWithMinLoss(EarlyStopping):
    """EarlyStoppingWithMinLoss extends the EarlyStopping callback to
    add a min loss value which will cause the model to exit"""
    def __init__(self, min_loss):
        # Hard coded values. Go directly to Jail. Do not pass go...
        EarlyStopping.__init__(self, 'loss', 0.000001, 100)
        self.min_loss = min_loss

    def on_epoch_end(self, epoch, logs=None):
        """This code is taken directly from the Keras Early stopping Callback.
        It simply has the minimum loss added"""
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        if current <= self.min_loss:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

class ReportWriter(object):
    """Class to write out a final report for the model parameters and performance"""
    def __init__(self, results, options, data):
        self._results = results
        self._options = options
        self._data = data
        self._class_names = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine',
            3: 'Ponderosa Pine',
            4: 'Cottenwood/Willow',
            5: 'Aspen',
            6: 'Douglas-fir',
            7: 'Krummholz'
        }

    def create(self, path):
        # open the reporting file
        # write out the details of the parameters for the report
        # (including weights etc)
        # write out the results of the training/testing
        self._open_report(path)
        self._training_details()
        self._describe_results()
        self._close_report()

    def _open_report(self, path):
        self._report = open(path, 'w')

    def _close_report(self):
        self._report.close()

    def _write_line(self, line):
        self._report.write('{}\n'.format(line))

    def _section_heading(self, title):
        self._write_line("================================================================================")
        self._write_line(title)
        self._write_line("================================================================================")

    def _title_and_value(self, title, value):
        self._write_line('{}: {}'.format(title, value))

    def _training_details(self):
        self._section_heading("TRAINING OPTIONS")
        self._title_and_value('Type', self._options.typ)
        self._title_and_value('Epochs', self._options.epochs)

        resample = self._options.resample if self._options.resample is not None else 'NA'
        self._title_and_value('Resample', resample)
        if self._options.resample is not None:
            for n in range(1, 8):
                _, labels = self._data.filter_by_category(self._data.x_train, self._data.y_train, n)
                self._title_and_value(self._class_names[n], str(labels.shape[0]))

        dropout = self._options.dropout if self._options.typ == 'modern' else 'NA'
        self._title_and_value('Dropout', dropout)

        layers = ", ".join(map(lambda x: str(x), self._options.layers)) if self._options.typ == 'modern' else 'NA'
        self._title_and_value('Layers', layers)

        self._title_and_value('Used Category Weights?', self._options.weight_data)
        if self._options.weight_data:
            weights = self._data.compute_class_weight()
            for n in range(7):
                self._title_and_value(self._class_names[n + 1], weights[n])

    def _describe_results(self):
        self._section_heading('RESULTS')
        self._title_and_value('Accuracy', self._results.accuracy)
        self._title_and_value('Weighted Accuracy', self._results.weighted_accuracy)
        self._write_line('')
        for n in range(1, 8):
            self._title_and_value(self._class_names[n], self._results.categorical_results[n])


# def write_model(model):
#     """Save the model to the output directory"""
#     model.save('/output/model.h5')

# def write_report_line(report, template, param):
#     """Write a single report line"""
#     report.write(template.format(param))

# def write_report(options, train_result, test_result, x_test, y_test, cat_weights, model):
#     """Write out the report on the model performance"""
#     final_training_loss, final_training_accuracy = train_result
#     final_testing_loss, final_testing_accuracy = test_result

#     typ = options['typ']
#     epochs = options['epochs']
#     dropout = options['dropout']
#     layers = options['layers']
#     report = open('/output/report.txt', 'w')

#     report.write("================================================================================\n")
#     report.write("SAMPLE WEIGHTS BY CATEGORY\n")
#     report.write("================================================================================\n")
#     for k in cat_weights.keys():
#         report.write("Category: {} Weight: {}\n".format(k, cat_weights[k]))
#     report.write("\n\n")
#     report.write("================================================================================\n")
#     report.write("OVERALL EVALUATION\n")
#     report.write("================================================================================\n")
#     write_report_line(report, 'Type: {}\n', typ)
#     write_report_line(report, 'Epochs: {}\n', epochs)
#     if typ == 'modern':
#         write_report_line(report, 'Dropout: {}\n', dropout)
#         write_report_line(report, 'Layers: {}\n', layers)
#     write_report_line(report, 'Final Training Loss: {}\n', final_training_loss)
#     write_report_line(report, 'Final Training Accuracy {}\n', final_training_accuracy)
#     write_report_line(report, 'Final Testing Loss: {}\n', final_testing_loss)
#     write_report_line(report, 'Final Testing Accuracy {}\n', final_testing_accuracy)

#     features_and_labels = np.hstack((x_test, y_test))

#     report.write("\n\n")
#     report.write("================================================================================\n")
#     report.write("PER CATEGORY EVALUATION\n")
#     report.write("================================================================================\n")
#     report.write(format_category_evaluation("Spruce/Fir", 1, features_and_labels, model))
#     report.write(format_category_evaluation("Lodgepole Pine", 2, features_and_labels, model))
#     report.write(format_category_evaluation("Ponderosa Pine", 3, features_and_labels, model))
#     report.write(format_category_evaluation("Cottenwood/Willow", 4, features_and_labels, model))
#     report.write(format_category_evaluation("Aspen", 5, features_and_labels, model))
#     report.write(format_category_evaluation("Douglas-fir", 6, features_and_labels, model))
#     report.write(format_category_evaluation("Krummholz", 7, features_and_labels, model))
#     report.close()

# def filter_by_category(features_labels, category):
#     """Return features and labels for a single category"""
#     filtered = np.array([r for r in features_labels if r[53 + category] == 1])
#     return filtered[...,:54], filtered[...,54:]

# def evaluate_by_category(features_labels, category, model):
#     """Perform an evaluation on the model for a single category"""
#     filtered_features, filtered_labels = filter_by_category(features_labels, category)
#     result = evaluate_model(model, filtered_features, filtered_labels)
#     return result

# def format_category_evaluation(name, category, features_labels, model):
#     """Return a formetted description of the category evaluation"""
#     [loss, accuracy] = evaluate_by_category(
#     features_labels, category, model)
#     return "{}: Loss: {}, Accuracy: {}\n".format(name, loss, accuracy)

# def filter_frame_by_category(frame, category):
#     return frame.loc[(frame['Cover_type'] == category)]

# def calculate_category_totals(frame):
#     return {cat: filter_frame_by_category(frame, cat).shape[0] for cat in range(1,8)}

# def max_category_total(cat_totals):
#     max_total = 0
#     for t in cat_totals.values():
#         max_total = max(max_total, t)
#     return max_total

# def calculate_category_weights(frame):
#     cat_totals = calculate_category_totals(frame)
#     max_total = max_category_total(cat_totals)
#     # scale the weights to try to balance the accuracy across different categories
#     return {cat: max(0.3 / (float(cat_totals[cat]) / float(max_total)), 1.0) for cat in range(1,8)}

# def calculate_sample_weight(cat_weights, labels):
#     sample_weights = []
#     for label in labels:
#         weights = [cat_weights[ind+1] * float(v) for ind, v in enumerate(label)]
#         sample_weights.append(sum(weights))
#     return np.array(sample_weights)

# def add_layer(model, n_units, dropout_rate, input_dim=None):
#     """Add a dense layer + a dropout layer to the model"""

#     if input_dim is not None:
#         model.add(Dense(n_units, activation='relu', input_dim=input_dim))
#     else:
#         model.add(Dense(n_units, activation='relu'))

#     model.add(Dropout(dropout_rate))

# def train_model(model, x_train, y_train, sample_weight, epochs):
#     """Train the given model using the given training data"""
#     epoch_logger = EpochEndCallback()
#     model.fit(x_train, y_train,
#               epochs=epochs,
#               batch_size=128,
#               verbose=0,
#               sample_weight=sample_weight,
#               callbacks=[epoch_logger])
#     return model

# def evaluate_model(model, x, y):
#     """Evaluate the given model using the given data"""
#     return model.evaluate(
#         x,
#         y,
#         batch_size=128,
#         verbose=0)

# def create_modern_model(parameters, x_shape):
#     """Create a model using modern parameters, with an architecture dictated by pattern.
#     Pattern should be an array of (dimensions, dropout_rate) tuples, one for each hidden layer"""

#     model = Sequential()
#     is_first_layer = True
#     for n, d in parameters:
#         if is_first_layer:
#             add_layer(model, n, d, x_shape[1])
#             is_first_layer=False
#         else:
#             add_layer(model, n, d)

#     model.add(Dense(7, activation='softmax'))

#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=sgd,
#                   metrics=['accuracy', f1])
#     return model

# def modern_train_test(parameters, x_train, x_test, y_train, y_test, sample_weight, epochs=20):
#     """Create, train and test a model using modern parameters, architected according to [parameters].
#     See docs for create_modern model for [parameters] explaination"""

#     model = create_modern_model(parameters, x_train.shape)
#     model = train_model(model, x_train, y_train, sample_weight, epochs)

#     final_train = evaluate_model(
#         model,
#         x_train,
#         y_train)

#     final_test = evaluate_model(
#         model,
#         x_test,
#         y_test)

#     return [final_train, final_test, model]

# def create_original_model(x_shape):
#     """To match the original paper:
#     activation='sigmoid'
#     kernal_initializer=TruncatedNormal(stddev=1.0)
#     hidden layer and output layer are 'sigmoid'
#     loss function is 'mean_squared_error'
#     """
#     model = Sequential()
#     model.add(Dense(120, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=1.0), input_dim=x_shape[1]))
#     model.add(Dense(7, activation='sigmoid'))

#     sgd = SGD(lr=0.05, decay=1e-6, momentum=0.5)
#     model.compile(loss='mean_squared_error',
#                   optimizer=sgd,
#                   metrics=['accuracy', f1])
#     return model

# def original_train_test(x_train, x_test, y_train, y_test, sample_weights, epochs=20):
#     """Create a model similar to that in the original paper.
#     Train it and then evaluate it"""

#     model = create_original_model(x_train.shape)
#     model = train_model(model, x_train, y_train, sample_weights, epochs)

#     final_train = evaluate_model(
#         model,
#         x_train,
#         y_train)

#     final_test = evaluate_model(
#         model,
#         x_test,
#         y_test)

#     return [final_train, final_test, model]

# def create_train_test_data(features, labels, resample):
#     """Split the data in to train and test (using stratification) amd convert the labels to keras categories.
#     returns (x_train, x_test, y_train, y_test) tuple"""
#     # Split the 'features' and 'income' data into training and testing sets
#     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

#     "Use Random Oversampling on the training data to address skew"
#     if resample:
#         x_train, y_train = RandomUnderSampler(random_state=RANDOM_SEED).fit_sample(x_train, y_train)

#     y_train -= 1 # change to 0 index
#     y_train = keras.utils.to_categorical(y_train, num_classes=7)
#     y_test -= 1
#     y_test = keras.utils.to_categorical(y_test, num_classes=7)

#     return x_train, x_test, y_train, y_test

def train_and_test(options):
    """"train and test the target models"""
    handler = create_data_handler(options)
    model = create_model_maker(options, handler.features.shape).create_model()
    trainer = Trainer(options, handler, model)
    trainer.train()
    tester = Tester(handler, model)
    results = tester.test()

    model.save('/output/{}-model.h5'.format(options.describe()))
    trainer.epoch_logger.save('/output/{}-logs.h5'.format(options.describe()))
    ReportWriter(results, options, handler).create('/output/{}-report.txt'.format(options.describe()))

    print("Loss", results.loss)
    print("Accuracy", results.accuracy)
    print("Weighted Accuracy", results.weighted_accuracy)
    print("Categorical Accuracy", results.categorical_results)
    # TODO replace this with a Data object with the frame, features and labels?
    #frame = load_data()
    #features, labels = prepared_data(frame)

    # TODO replace these with a TrainTestData object with x_train/test y_train/test, cat_weights, and sample weights.
    # should also include over/undersampling effects? Should also include y label -1 fixes and utils_to_categorical?
    # x_train, x_test, y_train, y_test = create_train_test_data(features, labels, options['resample'])
    # cat_weights = calculate_category_weights(frame)
    # sample_weight = calculate_sample_weight(cat_weights, y_train)

    #x_train, x_test, y_train, y_test = handler.create_train_test_data()






    # TODO replace this with a TrainingObject (use an interface, return concrete original or modern)
    # TODO returna results object including accuracy, loss, poss f1, per cat accuracy, weighted accuracy
    # sample_weight = None
    # cat_weights = []
    # if options.typ == 'original':
    #     train_result, test_result, model = original_train_test(x_train, x_test, y_train, y_test, handler.compute_sample_weight(), options.epochs)
    # else:
    #     architecture = [(x, options.dropout) for x in options.layers]
    #     train_result, test_result, model = modern_train_test(architecture, x_train, x_test, y_train, y_test, handler.compute_sample_weight(), options.epochs)

    # write_model(model)
    # write_report(options, train_result, test_result, x_test, y_test, cat_weights, model)

def parse_args():
    """Parse command line options into an options object via an options builder"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=['modern', 'original'], required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-r', '--resample', choices=['oversample', 'undersample'], default=None)
    parser.add_argument('-d', '--dropout', type=float, default=0.5)
    parser.add_argument('-l', '--layers', nargs='+', type=int, default=[])
    parser.add_argument('-n', '--num_per_class', nargs='+', type=int, default=[])
    parser.add_argument('-w', '--weight_data', action='store_true', default=False)
    parser.add_argument('-b', '--boost_per_class', nargs="+", type=float, default=[])
    parser.add_argument('-s', '--seed', type=int, default=randint(0, int(pow(2, 32)) -1))
    parser.add_argument('-m', '--min_loss', type=float, default=0.05)
    args = parser.parse_args()
    options = OptionBuilder().\
        typ(args.type).\
        epochs(args.epochs).\
        resample(args.resample).\
        dropout(args.dropout).\
        layers(args.layers).\
        num_per_class(args.num_per_class).\
        weight_data(args.weight_data).\
        boost_per_class(args.boost_per_class).\
        seed(args.seed).\
        min_loss(args.min_loss).\
        options()

    return options

if __name__ == "__main__":
    options = parse_args()
    print("USING SEED:", options.seed)
    np.random.seed(options.seed)
    train_and_test(options)

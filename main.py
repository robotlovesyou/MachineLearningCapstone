import sys
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

EPOCHS = 20

# List of all data names so that columns can be addressed by name
NAMES = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
        'Widerness_Area_Rawah', 'Widerness_Area_Neota', 'Widerness_Area_Comanche',
        'Widerness_Area_Cache_la_Poudre', 'Soil_type_2702', 'Soil_type_2703', 'Soil_type_2704',
        'Soil_type_2705', 'Soil_type_2706', 'Soil_type_2717', 'Soil_type_3501', 'Soil_type_3502',
        'Soil_type_4201', 'Soil_type_4703', 'Soil_type_4704', 'Soil_type_4744', 'Soil_type_4758',
        'Soil_type_5101', 'Soil_type_5151', 'Soil_type_6101', 'Soil_type_6102', 'Soil_type_6731',
        'Soil_type_7101', 'Soil_type_7102', 'Soil_type_7103', 'Soil_type_7201', 'Soil_type_7202',
        'Soil_type_7700', 'Soil_type_7701', 'Soil_type_7702', 'Soil_type_7709', 'Soil_type_7710',
        'Soil_type_7745', 'Soil_type_7746', 'Soil_type_7755', 'Soil_type_7756', 'Soil_type_7757',
        'Soil_type_7790', 'Soil_type_8703', 'Soil_type_8707', 'Soil_type_8708', 'Soil_type_8771',
        'Soil_type_8772', 'Soil_type_8776', 'Cover_type']

# List of all numerical data names which will have MinMax scaling applied
NUMERICAL = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points']

def write_report_line(report, template, param):
    report.write(template.format(param))

def write_report(options, results):
    [(final_training_loss, final_training_accuracy),
        (final_testing_loss, final_testing_accuracy)
    ] = results

    typ = options['typ']
    epochs = options['epochs']
    dropout = options['dropout']
    layers = options['layers']
    report = open('/output/report.txt', 'w')
    write_report_line(report, 'Type: {}\n', typ)
    write_report_line(report, 'Epochs: {}\n', epochs)
    if typ == 'modern':
        write_report_line(report, 'Dropout: {}\n', dropout)
        write_report_line(report, 'Layers: {}\n', layers)
    write_report_line(report, 'Final Training Loss: {}\n', final_training_loss)
    write_report_line(report, 'Final Training Accuracy {}\n', final_training_accuracy)
    write_report_line(report, 'Final Testing Loss: {}\n', final_testing_loss)
    write_report_line(report, 'Final Testing Accuracy {}\n', final_testing_accuracy)

def add_layer(model, n_units, dropout_rate, input_dim=None):
    """Add a dense layer + a dropout layer to the model"""

    if input_dim is not None:
        model.add(Dense(n_units, activation='relu', input_dim=input_dim))
    else:
        model.add(Dense(n_units, activation='relu'))

    model.add(Dropout(dropout_rate))

def create_modern_model(parameters, x_shape):
    """Create a model using modern parameters, with an architecture dictated by pattern.
    Pattern should be an array of (dimensions, dropout_rate) tuples, one for each hidden layer"""

    model = Sequential()
    is_first_layer = True
    for n, d in parameters:
        if is_first_layer:
            add_layer(model, n, d, x_shape[1])
            is_first_layer=False
        else:
            add_layer(model, n, d)

    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def modern_train_test(parameters, x_train, x_test, y_train, y_test, epochs=20):
    """Create, train and test a model using modern parameters, architected according to [parameters].
    See docs for create_modern model for [parameters] explaination"""

    model = create_modern_model(parameters, x_train.shape)

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=128,
              verbose=1)

    final_train = model.evaluate(
        x_train,
        y_train,
        batch_size=128,
        verbose=1)

    final_test = model.evaluate(
        x_test,
        y_test,
        batch_size=128,
        verbose=1)

    return [final_train, final_test]

def original_train_test(x_train, x_test, y_train, y_test, epochs=20):
    """To match the original paper:
    activation='sigmoid'
    kernal_initializer=TruncatedNormal(stddev=1.0)
    hidden layer and output layer are 'sigmoid'
    loss function is 'mean_squared_error'
    """

    model = Sequential()
    model.add(Dense(120, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=1.0), input_dim=x_train.shape[1]))
    model.add(Dense(7, activation='sigmoid'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.5)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=128,
              verbose=1)

    final_train = model.evaluate(
        x_train,
        y_train,
        batch_size=128,
        verbose=1)

    final_test = model.evaluate(
        x_test,
        y_test,
        batch_size=128,
        verbose=1)

    return [final_train, final_test]

def prepared_data():
    """Load the dataset from csv, scale the numerical data, convert the types and split the features and labels"""
    # Load the data
    df = pandas.read_csv('/dataset/covtype.data', header=None, names=NAMES)

    # Split the data into labels and features
    labels = df['Cover_type']
    features = df.drop('Cover_type', axis=1)

    # Scale all numerical features to ranges between 0 and 1
    scaler = MinMaxScaler()
    features[NUMERICAL] = scaler.fit_transform(features[NUMERICAL])

    labels = labels.values
    features = features.values.astype('float32')
    return features, labels


def create_train_test_data(features, labels):
    """Split the data in to train and test (using stratification) amd convert the labels to keras categories.
    returns (x_train, x_test, y_train, y_test) tuple"""
    # Split the 'features' and 'income' data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

    y_train -= 1 # change to 0 index
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_test -= 1
    y_test = keras.utils.to_categorical(y_test, num_classes=7)

    return x_train, x_test, y_train, y_test

def train_and_test(options):
    """"train and test the target models"""

    features, labels = prepared_data()
    x_train, x_test, y_train, y_test = create_train_test_data(features, labels)

    if options['typ'] == 'original':
        result = original_train_test(x_train, x_test, y_train, y_test, options['epochs'])
    else:
        architecture = [(x, options['dropout']) for x in options['layers']]
        result = modern_train_test(architecture, x_train, x_test, y_train, y_test, options['epochs'])

    write_report(options, result)

def parse_args():
    typ = ''
    epochs = 0
    dropout = 0.
    layers = []
    if sys.argv[1] == 'original':
        typ = 'original'
        epochs = int(sys.argv[2])
    else:
        typ = 'modern'
        epochs = int(sys.argv[2])
        dropout = float(sys.argv[3])
        layers = [int(arg) for arg in sys.argv[4:]]

    return {'typ': typ, 'epochs': epochs, 'dropout': dropout, 'layers': layers}

train_and_test(parse_args())

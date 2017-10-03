import sys
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from data import prepared_data

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

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

def train_model(model, x_train, y_train, epochs):
    """Train the given model using the given training data"""
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=128)
    return model

def evaluate_model(model, x, y):
    """Evaluate the given model using the given data"""
    return model.evaluate(
        x,
        y,
        batch_size=128)

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
    model = train_model(model, x_train, y_train, epochs)

    final_train = evaluate_model(
        model,
        x_train,
        y_train)

    final_test = evaluate_model(
        model,
        x_test,
        y_test)

    return [final_train, final_test]

def create_original_model(x_shape):
    """To match the original paper:
    activation='sigmoid'
    kernal_initializer=TruncatedNormal(stddev=1.0)
    hidden layer and output layer are 'sigmoid'
    loss function is 'mean_squared_error'
    """
    model = Sequential()
    model.add(Dense(120, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=1.0), input_dim=x_shape[1]))
    model.add(Dense(7, activation='sigmoid'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.5)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def original_train_test(x_train, x_test, y_train, y_test, epochs=20):
    """Create a model similar to that in the original paper.
    Train it and then evaluate it"""

    model = create_original_model(x_train.shape)
    model = train_model(model, x_train, y_train, epochs)

    final_train = evaluate_model(
        model,
        x_train,
        y_train)

    final_test = evaluate_model(
        model,
        x_test,
        y_test)

    return [final_train, final_test]

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
    """parse command line args to create run options"""
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

if __name__ == "__main__":
    train_and_test(parse_args())

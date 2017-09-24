import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# List of all data names so that columns can be addressed by name
names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Widerness_Area_Rawah', 'Widerness_Area_Neota', 'Widerness_Area_Comanche',
    'Widerness_Area_Cache_la_Poudre', 'Soil_type_2702', 'Soil_type_2703', 'Soil_type_2704', 'Soil_type_2705',
    'Soil_type_2706', 'Soil_type_2717', 'Soil_type_3501', 'Soil_type_3502', 'Soil_type_4201', 'Soil_type_4703',
    'Soil_type_4704', 'Soil_type_4744', 'Soil_type_4758', 'Soil_type_5101', 'Soil_type_5151', 'Soil_type_6101',
    'Soil_type_6102', 'Soil_type_6731', 'Soil_type_7101', 'Soil_type_7102', 'Soil_type_7103', 'Soil_type_7201',
    'Soil_type_7202', 'Soil_type_7700', 'Soil_type_7701', 'Soil_type_7702', 'Soil_type_7709', 'Soil_type_7710',
    'Soil_type_7745', 'Soil_type_7746', 'Soil_type_7755', 'Soil_type_7756', 'Soil_type_7757', 'Soil_type_7790',
    'Soil_type_8703', 'Soil_type_8707', 'Soil_type_8708', 'Soil_type_8771', 'Soil_type_8772', 'Soil_type_8776',
    'Cover_type']

# List of all numerical data names which will have MinMax scaling applied
numerical = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points']

def default_train_test(x_train, x_test, y_train, y_test, epochs=20):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(240, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=128,
              verbose=1)

    return model.evaluate(
        x_test,
        y_test,
        batch_size=128,
        verbose=1)

def original_train_test(x_train, x_test, y_train, y_test, epochs=20):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(120, activation='sigmoid', input_dim=x_train.shape[1]))
    model.add(Dense(7, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=128,
              verbose=1)

    return model.evaluate(
        x_test,
        y_test,
        batch_size=128,
        verbose=1)

# Load the data
df = pandas.read_csv('./dataset/covtype.data', header=None, names=names)

# Split the data into labels and features
labels = df['Cover_type']
features = df.drop('Cover_type', axis=1)

# Scale all numerical features to ranges between 0 and 1
scaler = MinMaxScaler()
features[numerical] = scaler.fit_transform(features[numerical])

labels = labels.values
features = features.values.astype('float32')

random_seed = 1234
np.random.seed(random_seed)

# Split the 'features' and 'income' data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)

y_train -= 1 # change to 0 index
y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_test -= 1
y_test = keras.utils.to_categorical(y_test, num_classes=7)

print("Original method", original_train_test(x_train, x_test, y_train, y_test, 100))
print("Final score", default_train_test(x_train, x_test, y_train, y_test, 100))
from sklearn.preprocessing import MinMaxScaler
import pandas

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

def load_data():
    """Load the dataset into a pandas dataframe"""
    return pandas.read_csv('/dataset/covtype.data', header=None, names=NAMES)

def prepared_data():
    """Load the dataset from csv, scale the numerical data, convert the types and split the features and labels"""
    # Load the data
    df = load_data()

    # Split the data into labels and features
    labels = df['Cover_type']
    features = df.drop('Cover_type', axis=1)

    # Scale all numerical features to ranges between 0 and 1
    scaler = MinMaxScaler()
    features[NUMERICAL] = scaler.fit_transform(features[NUMERICAL])

    labels = labels.values
    features = features.values.astype('float32')
    return features, labels
import numpy as np
from main import create_modern_model, create_original_model, train_model, evaluate_model, prepared_data, create_train_test_data
# create a copy of the original model and the new model.
# train them on the training data
# evaluate the models against all the testing data
# split the testing data by category.
# evaluate the models against each category
EPOCHS=1
features, labels = data = prepared_data()
x_train, x_test, y_train, y_test = create_train_test_data(features, labels)

original_model = create_original_model(x_train.shape)
modern_model = create_modern_model([(10, 0.5), (10, 0.5), (10, 0.5)], x_train.shape)

# original_model = train_model(original_model, x_train, y_train, EPOCHS)
# modern_model = train_model(modern_model, x_train, y_train, EPOCHS)

# original_result = evaluate_model(original_model, x_test, y_test)
# modern_result = evaluate_model(modern_model, x_test, y_test)
print("test shape", y_test.shape)
recombined_data = np.hstack((x_test, y_test))
print("shape", recombined_data.shape)

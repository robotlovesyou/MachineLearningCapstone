import numpy as np
from keras.models import load_model
from main import create_modern_model, create_original_model, train_model, evaluate_model, prepared_data, create_train_test_data
# create a copy of the original model and the new model.
# train them on the training data
# evaluate the models against all the testing data
# split the testing data by category.
# evaluate the models against each category

def filter_by_category(features_labels, category):
  filtered = np.array([r for r in features_labels if r[53 + category] == 1])
  return filtered[...,:54], filtered[...,54:]

def evaluate_by_category(features_labels, category, model):
  filtered_features, filtered_labels = filter_by_category(features_labels, category)
  result = evaluate_model(model, filtered_features, filtered_labels)
  return result

def format_category_evaluation(name, category, features_labels, model):
  [loss, accuracy] = evaluate_by_category(
    features_labels, category, model)

  return "{}: Loss: {}, Accuracy: {}\n".format(name, loss, accuracy)

features, labels = data = prepared_data()
x_train, x_test, y_train, y_test = create_train_test_data(features, labels)

model = load_model('/model/model.h5')

result = evaluate_model(model, x_test, y_test)
report_file = open('/output/final_report.txt', 'w')
report_file.write("Overall Result - Loss: {} Accuracy {}\n".format(result[0], result[1]))

features_and_labels = np.hstack((x_test, y_test))

report_file.write(format_category_evaluation("Spruce/Fir", 1, features_and_labels, model))
report_file.write(format_category_evaluation("Lodgepole Pine", 2, features_and_labels, model))
report_file.write(format_category_evaluation("Ponderosa Pine", 3, features_and_labels, model))
report_file.write(format_category_evaluation("Cottenwood/Willow", 4, features_and_labels, model))
report_file.write(format_category_evaluation("Aspen", 5, features_and_labels, model))
report_file.write(format_category_evaluation("Douglas-fir", 6, features_and_labels, model))
report_file.write(format_category_evaluation("Krummholz", 7, features_and_labels, model))
report_file.close()

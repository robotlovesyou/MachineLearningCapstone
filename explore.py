from data import load_data

def filter_by_category(frame, category):
  return frame.loc[(frame['Cover_type'] == category)]

def describe_category(frame, filtered_frame, name):
  total_rows = frame.shape[0]
  filtered_rows = filtered_frame.shape[0]
  percentage = (filtered_rows / total_rows)
  print("There are {} instances of {} representing {:.2%} of the total".format(
    filtered_rows, name, percentage
  ))

def filter_and_describe_category(frame, category, name):
  describe_category(frame, filter_by_category(frame, category), name)

df = load_data('dataset/covtype.data')

print("The data has {} rows".format(df.shape[0]))

filter_and_describe_category(df, 1, 'Spruce/Fir')
filter_and_describe_category(df, 2, 'Lodgepole Pine')
filter_and_describe_category(df, 3, 'Ponderosa Pine')
filter_and_describe_category(df, 4, 'Cottonwood/Willow')
filter_and_describe_category(df, 5, 'Aspen')
filter_and_describe_category(df, 6, 'Douglas-fir')
filter_and_describe_category(df, 7, 'Krummholz')
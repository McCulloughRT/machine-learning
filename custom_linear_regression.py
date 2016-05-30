import numpy as np
from math import sqrt

def get_numpy_data(data_sframe, features, output):
  data_sframe['constant'] = 1
  features = ['constant'] + features
  features_sframe = data_sframe[features]
  feature_matrix = features_sframe.to_numpy()
  output_sarray = data_sframe['price']
  output_array = output_sarray.to_numpy()
  return(feature_matrix, output_array)

def predict_output(feature_matrix, weights):
  predictions = np.dot(feature_matrix, weights)
  return(predictions)

def feature_derivative(errors, feature):
  derivative = 2*(np.dot(errors, feature))
  return(derivative)

def regression_gradient_descent(train_data, features, output, initial_weights, step_size, tolerance):
  (feature_matrix, output) = get_numpy_data(train_data, features, output)
  converged = False
  weights = np.array(initial_weights)
  while not converged:
    predictions = predict_output(feature_matrix, weights)
    errors = predictions - output
    gradient_sum_squares = 0

    for i in range(len(weights)):
      derivative_i = feature_derivative(errors, feature_matrix[:,i])
      gradient_sum_squares = gradient_sum_squares + (derivative_i**2)
      weights[i] = weights[i] - (step_size * derivative_i)

    gradient_magnitude = sqrt(gradient_sum_squares)
    if gradient_magnitude < tolerance:
      converged = True
  return(weights)
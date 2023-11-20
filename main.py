import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([0, 30, 45, 10, 55, 70, 95, 60]) # X is the input data
y = np.array([0, 0, 0, 0, 1, 1, 1, 1]) # y is the desired output

# Algorithm parameters (parameters are the numbers that we want to learn, actual ML)
w1 = 0.2 # w1 is a weight that we are trying to learn the value of
w0 = -0.25 # w0 is a weight that we are trying to learn the value of

# Hyperparameters (Numbers we can change to control the running of the algorithm) they control the training/learning/fitting
lr = 0.005
num_epochs = 20000 # num of times it will loop over input data

# Empty list that stores the value of e, eachtime it runs over the loop
E_values = []  # Empty list to store the values of e

for j in range(num_epochs): # Is a loop that starts at 0 and goes to num_epochs - 1, looping over num_epochs amount of times
    for i in range(len(X)): # creates a loop where each x input value is tested within the model. We did this by using the function len which puts down num of x values
        y_hat = 1 / (1 + np.exp(-(w0 + w1*X[i])))  # compute output of neural network
        e = 0.5 * (y_hat - y[i]) ** 2 # e is short for error and it compares the output we got with the initial y value, when .5 is multiplied by 2 because of power rule, it cancels out
        E_values.append(e)  # Append the current value of e to the list

        w1 = w1 - lr * (y_hat - y[i]) * y_hat * (1 - y_hat) * X[i] # This takes the inital values of w1 and replaces it with the new values of w1, using gradient descent
        w0 = w0 - lr * (y_hat - y[i]) * y_hat * (1 - y_hat) # This takes the inital values of w0 and replaces it with the new values of w0, using gradient descent
    print(f"epochs: {j}, Example: {i}, e: {e}, w1: {w1}, w0: {w0}")

# Plotting the values of E
plt.plot(range(len(E_values)), E_values) 
plt.xlabel('Index of every input for every epoch')
plt.ylabel('Error')
plt.title('Error at every input for every epoch')
plt.show()

# Testing 100 grade values to see how acurate our model is
highest_grade = 100
out = [] # This is a list which holds highest_grade output values
for grade in range(highest_grade + 1):
  y_hat = 1 / (1 + np.exp(-(w0 + w1*grade)))  # compute output of neural network
  print(y_hat)
  out.append(y_hat)
plt.plot(range(highest_grade + 1), out)

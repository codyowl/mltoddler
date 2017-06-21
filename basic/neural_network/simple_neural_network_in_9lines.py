import numpy as np
#exp is the 'exponential'
first_input = [0,0,1]
second_input = [1,1,1]
third_input = [1,0,1]
fourth_input = [0,1,1]

#train_output
output = [0,1,1,0]

training_set_inputs = np.array([first_input, second_input,third_input,fourth_input])
training_set_outputs = np.array([first_output]).T 

np.random.seed(1)
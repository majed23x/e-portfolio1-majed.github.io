import numpy as np



inputs = np.array([45, 25])
type(inputs)
inputs[0]
weights = np.array([0.7, 0.1])
weights[0]


# Create Sum Function
def sum_func(inputs, weights):
    return inputs.dot(weights)

s_prob1 = sum_func(inputs, weights)
s_prob1 



def step_function(sum_func):
  if (sum_func >= 1):
    print(f'The Sum Function is greater than or equal to 1')
    return 1
  else:
        print(f'The Sum Function is NOT greater')
        return 0
  


step_function(s_prob1 )


weights = [-0.7, 0.1]


s_prob2 = sum_func(inputs, weights)

round(s_prob2, 2)

step_function(s_prob2 )
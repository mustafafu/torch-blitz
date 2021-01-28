# Imports
import torch
import numpy as np

#Initializing tensors

# Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# From numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)



#From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


#With shape declaration
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



#Tensor Attributes
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")



#Tensor Operations

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

#Numpy indexing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)



#Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#Multiplyinh Tensors
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# This computes the matrix multiplication between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

#Inplace operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)
#Warning:In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.



#Bridging back and forth to NUMPY
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
#A change in tensor reflects in the Numpy array !!!!!!!
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#Numpy to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
#Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

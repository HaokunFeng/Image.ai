import torch
import tensorflow as tf
flag = torch.cuda.is_available()
if flag:
    print("CUDA is available")
else:
    print("CUDA is not available")
ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Driver: ",device)
print("GPU: ",torch.cuda.get_device_name(0))
print(tf.__version__)
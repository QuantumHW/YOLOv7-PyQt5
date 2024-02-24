import torch

print('pyTorch版本:'+torch.__version__)
print('CUDA是否可用:'+str(torch.cuda.is_available()))
print('GPU设备个数:'+str(torch.cuda.device_count()))
print('cuDnn版本号:'+str(torch.backends.cudnn.version()))
print('CUDA版本号:'+str(torch.version.cuda))
from thop import profile
import torch
def find_flops(model, input = -1):
    '''
        The input needs to be a PyTorch model and it will return the number
        of flops and parameters of the model.
        The dataset is considered to be CIFAR10 and the batch size is 1.
    '''
    if type(input) == int:
        input = torch.randn(1, 3, 32, 32)

    macs, params = profile(model, inputs=(input, ))
    print(f"Flops of the model: {macs}; Parameters of the model:{params}")
    return macs,params
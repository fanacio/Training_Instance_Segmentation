import torch
import os
import cv2
from yolact import Yolact

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = 'pth_model/yolact_base_505_100000.pth'
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net.to(device)

    #计算模型参数总量
    total=sum([param.nelement() for param in net.parameters()])
    print("Total of parameters: %.2fM" %(total/1e6))

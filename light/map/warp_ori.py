import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as dgm
H = [[1.86778852, 1.95464715e-02, -8.54657426e+02],
     [-2.29551386e-03, 1.86696288, -4.94259414e+02],
     [1.02875633e-05, 1.04192777e-05, 9.83393872e-01]]
T = [[2/1920, 0 ,-1],[0,2/1208,-1],[0,0,1]]
HT_inv = np.matmul(np.array(H), np.linalg.inv(np.array(T)))
THT_inv = np.matmul(T,HT_inv)
class MyHomography(nn.Module):
    def __init__(self, direc):
        super(MyHomography, self).__init__()
        assert (direc == '120_60' or direc == '60_120') 
        if direc == '120_60':
            self.homo = np.linalg.inv(THT_inv)
        elif direc == '60_120':
            self.homo = THT_inv
        else:
            raise("error: argument 'direc' is not in ['120_60','60_120'], there are only two choices")
    def forward(self,device,N):
        #return torch.unsqueeze(self.homo, dim=0).to(device) # 1x3x3
        homo_NHW = np.empty((N, *self.homo.shape), dtype = float)
        for i in range(N):
            homo_NHW[i] = self.homo        
        return torch.Tensor(homo_NHW).to(device)
def warp_kornia(img_src, direc):
    """
    :param img_src: NCHW  tensor(cv2.imread)   255
    :return: NCHW  tensor  normal
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #img_src_tensor =   .utils.image_to_tensor(img_src).float() / 255. #NCHW    
    img_src_tensor = img_src / 255.
    img_src_tensor = img_src_tensor.to(device)
    dst_homo_src = MyHomography(direc).to(device)


    height, width = img_src_tensor.shape[-2:]
    warper = dgm.HomographyWarper(height, width)

    warped_img_tensor = 255. * warper(img_src_tensor, dst_homo_src(device,img_src_tensor.shape[0]))
    warped_img = warped_img_tensor.cpu().detach().numpy()
    #warped_img = dgm.utils.tensor_to_image(warped_img_tensor)
    #print(warped_img_tensor.shape)
    return warped_img_tensor
def warp_kornia_single(img_src):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_src_tensor = dgm.utils.image_to_tensor(img_src).float() / 255.
    img_src_tensor = img_src_tensor.view(1, *img_src_tensor.shape)  # 1xCxHxW

    img_src_tensor = img_src_tensor.to(device)
    dst_homo_src = MyHomography().to(device)


    height, width = img_src_tensor.shape[-2:]
    warper = dgm.HomographyWarper(height, width)

    warped_img_tensor = 255. * warper(img_src_tensor, dst_homo_src(device,img_src_tensor.shape[0]))
    warped_img = dgm.utils.tensor_to_image(warped_img_tensor)
    #print(warped_img_tensor.shape)
    return warped_img
def warp_cv_single(img_src):
    Homography_array = np.array(H)
    height, width = img_src.shape[-3:-1]
    img_warped = cv2.warpPerspective(img, Homography_array, (width, height), flags=cv2.INTER_NEAREST)
    return img_warped
if __name__ == "__main__":
    img = cv2.imread('./120_60/6-120-10125.jpg', cv2.IMREAD_COLOR)
    #print(img,img.shape)
    cv2.imwrite('warpedcv.png', warp_kornia_single(img))
    print('Finished!')

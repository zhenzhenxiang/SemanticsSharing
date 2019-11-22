import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as dgm    
import math
    
class warp_kornia(nn.Module):
    """
    :param img_src: NCHW  tensor(cv2.imread)   255
    :return: NCHW  tensor  normal
    """
    def __init__(self):
        super(warp_kornia, self).__init__()  
             
                
    def forward(self, img_src, direc, scale):             
        height, width = img_src.shape[-2:]
        
        self.warper = dgm.HomographyWarper(height, width)
        self.MyHomography = Homography(direc, height, width, N=img_src.shape[0], scale=scale)  
        
        img_src_tensor = img_src / 255.        
        
        warped_img_tensor = 255. * self.warper(img_src_tensor, self.MyHomography())        
        return warped_img_tensor
    
    
class Homography(nn.Module):
    def __init__(self, direc, height, width, N, scale):
        super(Homography, self).__init__()
        
        self.N = N
        self.THT_inv = THT_inv_matrix(height, width, scale)
        self.homo_NHW = torch.Tensor(N,3,3) 
        
        assert (direc == '120_60' or direc == '60_120') 
        if direc == '120_60':
            self.homo = np.linalg.inv(self.THT_inv)
        elif direc == '60_120':
            self.homo = self.THT_inv
        else:
            raise("error: argument 'direc' is not in ['120_60','60_120'], there are only two choices")        
        
    def forward(self):
        for i in range(self.N):
            self.homo_NHW[i] = torch.tensor(self.homo)  
        
        return self.homo_NHW.cuda() #.view(self.N, *self.homo.shape).to('cuda')
    

def THT_inv_matrix(height, width, scale=1.0):
    def eulerAnglesToRotationMatrix(theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    K1 = np.array([[1796.6985372203135, 0., 982.1333137421029],
                   [0., 1793.8563682988065, 696.3526269889819],
                   [0., 0., 1.]])
    K2 = np.array([[967.1103975517259, 0., 967.1468712919049],
                   [0., 964.5298875990683, 629.2707542395874],
                   [0., 0., 1.]])
    S = np.array([[scale, scale, scale],
                  [scale, scale, scale],
                  [1., 1., 1.]])
    R = eulerAnglesToRotationMatrix((-0.01, 0.01, 0.005))

    H = np.asmatrix(K1 * S) * np.asmatrix(R.transpose()) * np.linalg.inv(np.asmatrix(K2 * S))   
    '''
    H1 = np.array[[1.86778852,      1.95464715e-02, -8.54657426e+02],
                 [-2.29551386e-03, 1.86696288, -4.94259414e+02],
                 [1.02875633e-05, 1.04192777e-05, 9.83393872e-01]]
    '''
    T = [[2 / width, 0, -1], [0, 2 / height, -1], [0, 0, 1]]

    HT_inv = np.matmul(H, np.linalg.inv(T))
    THT_inv = np.matmul(T, HT_inv)
    return THT_inv

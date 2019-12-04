"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union']


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            
            correct, labeled = batch_pix_accuracy(pred, label)# correct prediction number of pixel , number of pixel which needed predict 
            inter, union = batch_intersection_union(pred, label, self.nclass)# list there are 'nclass' element

            self.total_correct += correct
            self.total_label += labeled

            self.total_inter += inter.double() 
            self.total_union += union.double()
        for i in range(preds.shape[0]):
            evaluate_worker(self, preds[i].view(1,*preds[i].shape), labels[i].view(1,*labels[i].shape))
    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (1e-10 + self.total_label)  # remove np.spacing(1)        
        IoU = 1.0 * self.total_inter / (1e-10 + self.total_union)        
        mIoU = IoU.mean().item()
        return pixAcc, mIoU ,IoU
    
    
    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass).double()
        self.total_union = torch.zeros(self.nclass).double()
        self.total_correct = 0.0
        self.total_label = 0.0
        #####################################
        self.pix_acc_per_img = []
        self.iou_per_img = []
        self.miou_per_img = []
        
        self.class_num_tensor_for_batch = torch.zeros(self.nclass)
        ####################################

        



def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1
    
    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass

    predict = torch.argmax(output, 1).double()+ 1
    target = target.double() + 1

    predict = predict * (target > 0).double()
    intersection = predict * (predict == target).double()

    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi).double()
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi).double()
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi).double()
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        
    return area_inter.double(), area_union.double()




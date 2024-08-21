import torch

from typing import List

# own modules
from metrics.parent_metric import Metric



class IntersectOverUnion(Metric):
    
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes : List[str]) -> None:
        
        self.classes = classes
        
        # Number of classes
        self.num_classes = len(classes)
        self.class_list = [i for i in range(0, len(self.classes))]

        self.reset()

    
    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self._mIoU():.2f}"
            

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        # Arrays to accumulate intersection and union values for each class
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float32)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float32)


    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width. --> done
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels). --> done
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        # Get unique values in target
        unique_values = set(torch.unique(target).tolist())
        
        # in order to also be able to use the resnet with the same script 
        if 255 in unique_values:
            unique_values.remove(255)
        
        # Check the shapes and values
        if prediction.dim() != 4:
            raise ValueError("Prediction must have shape (b, c, h, w).")

        if target.shape[1:] != prediction.shape[2:] and target.dim() != 3:
            raise ValueError("Target must have shape (b, h, w) matching the prediction height and width.")

        # Ensure target values are within the range of classes and/or ignore index        
        if not unique_values.issubset(self.class_list):
            raise ValueError("Target values must be between 0 and c - 1.")
        
        
        # Obtain the predicted class for each pixel
        predicted_class = torch.argmax(prediction, dim=1)

        # Calculate IoU for each class
        for class_number in unique_values:
            
            # Calculate intersection and union for the current class
            pred_mask = (predicted_class == class_number)
            target_mask = (target == class_number)
            
            # Ignore pixels with value 255
            ignored_mask = (target == 255)
            pred_mask = pred_mask & ~ignored_mask
            target_mask = target_mask & ~ignored_mask
            
            # True positives
            intersection = (pred_mask & target_mask).sum().item()
            
            # False positives and false negatives
            false_positives = (pred_mask & ~target_mask).sum().item()
            false_negatives = (~pred_mask & target_mask).sum().item()

            # Union (sum of true positives, false positives, and false negatives)
            union = intersection + false_positives + false_negatives
            
            # Update intersections and unions for each class
            self.intersections[class_number] += intersection
            self.unions[class_number] += union
        
    
    def _mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
       
        # return 0 if no data is available
        if self.intersections.sum() == 0:
            return 0.0
        
        # Calculate IoU for each class
        iou = torch.zeros(len(self.classes), dtype=torch.float32)
        
        for idx in range(len(self.classes)):
        
            if self.unions[idx] == 0:
                # If the union is 0, set IoU to 0 to avoid division by zero
                iou[idx] = 0.0
            else:
                iou[idx] = self.intersections[idx] / self.unions[idx]

        # Calculate mean IoU across all classes        
        return iou.mean().item()
    
    
    def calculate_mIoU(self) -> float:
        '''
        Return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        return self._mIoU()



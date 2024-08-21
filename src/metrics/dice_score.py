import torch

from typing import List

# own modules
from metrics.parent_metric import Metric



class DiceScore(Metric):
    
    '''
    Dice Score
    '''

    def __init__(self, num_classes : int) -> None:
        
        # Number of classes
        self.num_classes = num_classes
        self.class_list = [i for i in range(0, self.num_classes)]

        self.reset()

    
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
        predicted_class = torch.argmax(prediction, dim = 1)
        
        # Calculate the Dice Score for each class
        for i in range(self.num_classes):
            
            # Ignore pixels with value 255
            if i == 255:
                continue
            
            # Calculate the intersection and union
            intersection = torch.logical_and(target == i, predicted_class == i).sum()
            union = torch.logical_or(target == i, predicted_class == i).sum()
            
            # Accumulate the intersection and union
            self.intersections[i] += intersection
            self.unions[i] += union
            
            
    def calculate_metric(self) -> float:
        
        if self.intersections.sum() == 0:
            return 0
        
        # Calculate the Dice Score 
        dice_score = (2 * self.intersections.sum() / self.unions.sum())
        
        # Return the mean Dice Score to get the overall dice coefficient for the image as a whole
        return dice_score.mean().item()
    
    
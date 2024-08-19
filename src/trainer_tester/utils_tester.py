import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import topk
from torch.nn import functional as F

class GradCAM:
    
    def __init__(self, model, test_loader, save_dir, device) -> None:
        self.save_dir = save_dir
        self.device = device
        self.model = model
        self.model.eval()

        self.features_blobs = []
        def hook_feature(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())
        self.model._modules.get('layer3').register_forward_hook(hook_feature)
        
        # Get the softmax weight
        params = list(self.model.parameters())
        self.weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
        self.test_loader = test_loader

    def process_batch(self, inputs, targets):
        # Forward pass through model
        outputs = self.model(inputs)
        # Get the softmax probabilities
        probs = F.softmax(outputs, dim=1).data
        # Get the class indices of top k probabilities
        class_idx = torch.argmax(probs, dim=1)
        
        for i in range(inputs.size(0)):
            image_tensor = inputs[i:i+1]
            prob = probs[i]
            idx = class_idx[i].item()

            # Generate class activation mapping for the top prediction
            feature_conv = self.features_blobs[0][i]  # Shape should be (C, H, W)
            CAMs = self.returnCAM(feature_conv, self.weight_softmax, [idx])
            orig_image = inputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
            orig_image = (orig_image - orig_image.min()) / (orig_image.max() - orig_image.min())  # Normalize

            # Prepare for visualization
            width, height = orig_image.shape[1], orig_image.shape[0]
            save_name = f"Test_{i}"
            self.show_cam(CAMs, width, height, orig_image, [idx], self.test_loader.dataset.classes, save_name)

    def run(self):
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device).long()
            self.process_batch(inputs, targets)

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # Generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        nc, h, w = feature_conv.shape  # Shape should be (C, H, W)
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def show_cam(self, CAMs, width, height, orig_image, class_idx, all_classes, save_name):
        for i, cam in enumerate(CAMs):
            heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + orig_image * 0.5
            # Put class label text on the result
            cv2.putText(result, all_classes[class_idx[i]], (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('CAM', result / 255.)
            cv2.waitKey(0)
            cv2.imwrite(f"{self.save_dir}/CAM_{save_name}.jpg", result)
    
    def _create_folder(self):
        # Create model save directory
        self.gradcam_save_dir = Path(f"{self.save_dir}/GradCAM")
        self.gradcam_save_dir.mkdir(exist_ok=True)
        
        # Create subfolder for the current test checkpoint        
        self.subfilter = Path(f"{self.gradcam_save_dir}/{self.test_checkpoint_type.split('.')[0]}")
        self.subfilter.mkdir(exist_ok=True)

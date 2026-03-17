import torch
from torchvision import datasets
import torchvision.transforms as transforms
import os
from PIL import Image
import cv2

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor(), train=True):
        super(CustomDataset, self).__init__()
        # Initialize dataset properties here
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Load your dataset files or directories here
        self.files = os.listdir(root_dir)  # List all files in the directory
        print(f"Loaded {len(self.files)} files.")  # Replace this with actual file loading logic
    
    def __len__(self):
        return len(self.files) // 10    # input + gt = 10 frames per scene
    
    def __getitem__(self, idx):
        """
        Returns:
            inputs (Tensor): A batch of 9 input images.
            target (Tensor): The corresponding ground truth image.
        """
        # Load the input and target images from disk or memory
        input_images = []

        if not self.train:
            idx = idx * 10
            names = os.listdir(self.root_dir)
            for i in range(9):  # Assuming you have 8 input images per sample
                img_path = f"{self.root_dir}{names[idx+i+1]}"  # Adjust path according to your naming convention
                img_tensor = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                input_images.append(img_tensor)
            
            target_img_path = f"{self.root_dir}{names[idx]}"  # Adjust path according to your naming convention

            target_tensor = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
            if self.transform:
                target_tensor = self.transform(target_tensor)
        else:
            for i in range(9):  # Assuming you have 8 input images per sample
                img_path = f"{self.root_dir}Scene-{idx:03}-in-{i}.tif"  # Adjust path according to your naming convention
                img_tensor = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                input_images.append(img_tensor)
        
            target_img_path = f"{self.root_dir}Scene-{idx:03}-gt.tif"  # Adjust path according to your naming convention

            target_tensor = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
            if self.transform:
                target_tensor = self.transform(target_tensor)
        
        inputs = torch.stack(input_images)  # Stack the input images into a single tensor
        target = target_tensor
        
        return inputs, target
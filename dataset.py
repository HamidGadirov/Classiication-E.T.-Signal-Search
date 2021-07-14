import numpy as np
import torch
from torch.utils.data import Dataset
# class ETIDataset:
#     def __init__(self,image_paths,targets):
#         self.image_paths = image_paths
#         self.targets = targets

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self,item):
#         image = np.load(self.image_paths[item]).astype(float)
#         target = self.targets[item]
#         image = image/np.array([np.abs(image).max() for i in range(6)]).reshape(6,1,1)
#         return {'image': torch.tensor(image,dtype=torch.float),
#                 'target': torch.tensor(target,dtype=torch.long)}
    
class ETIDataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image = np.load(self.images_filepaths[idx]).astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        
        if self.transform is not None:
            image = self.transform()(image=image)["image"]
        else:
            image = image[np.newaxis,:,:]
            image = torch.from_numpy(image).float()
        
        target = self.targets[idx]
        return {'image': torch.tensor(image,dtype=torch.float),
                'target': torch.tensor(target,dtype=torch.long)}

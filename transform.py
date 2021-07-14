import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2

#https://www.kaggle.com/ligtfeather/pytorch-lightning-grid-mask-ranger-opt-w-b

# class GridMask(DualTransform):

#     def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
#         super(GridMask, self).__init__(always_apply, p)
#         if isinstance(num_grid, int):
#             num_grid = (num_grid, num_grid)
#         if isinstance(rotate, int):
#             rotate = (-rotate, rotate)
#         self.num_grid = num_grid
#         self.fill_value = fill_value
#         self.rotate = rotate
#         self.mode = mode
#         self.masks = None
#         self.rand_h_max = []
#         self.rand_w_max = []

#     def init_masks(self, height, width):
#         if self.masks is None:
#             self.masks = []
#             n_masks = self.num_grid[1] - self.num_grid[0] + 1
#             for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
#                 grid_h = height / n_g
#                 grid_w = width / n_g
#                 this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
#                 for i in range(n_g + 1):
#                     for j in range(n_g + 1):
#                         this_mask[
#                              int(i * grid_h) : int(i * grid_h + grid_h / 2),
#                              int(j * grid_w) : int(j * grid_w + grid_w / 2)
#                         ] = self.fill_value
#                         if self.mode == 2:
#                             this_mask[
#                                  int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
#                                  int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
#                             ] = self.fill_value
                
#                 if self.mode == 1:
#                     this_mask = 1 - this_mask

#                 self.masks.append(this_mask)
#                 self.rand_h_max.append(grid_h)
#                 self.rand_w_max.append(grid_w)

#     def apply(self, image, mask, rand_h, rand_w, angle, **params):
#         h, w = image.shape[:2]
#         mask = AF.rotate(mask, angle) if self.rotate[1] > 0 else mask
#         mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
#         image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
#         return image

#     def get_params_dependent_on_targets(self, params):
#         img = params['image']
#         height, width = img.shape[:2]
#         self.init_masks(height, width)

#         mid = np.random.randint(len(self.masks))
#         mask = self.masks[mid]
#         rand_h = np.random.randint(self.rand_h_max[mid])
#         rand_w = np.random.randint(self.rand_w_max[mid])
#         angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

#         return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

#     @property
#     def targets_as_params(self):
#         return ['image']

#     def get_transform_init_args_names(self):
#         return ('num_grid', 'fill_value', 'rotate', 'mode')

#https://www.kaggle.com/fauzanalfariz/seti-e-t-efficientnet-b4-signal-detection
# class MixUpAugmentation():
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     def __init__(self, x, y, alpha, use_cuda):
#         self.x = x
#         self.y = y
#         self.alpha = alpha
#         self.use_cuda = use_cuda
        
#     def __getitem__(self):
#         if self.alpha > 0:
#             lmbda = np.random.beta(self.alpha, self.alpha)
#         else:
#             lmbda = 1
         
#         batch_size = self.x.size()[0]
#         if self.use_cuda:
#             index = torch.randperm(batch_size).cuda()
#         else:
#             index = torch.randperm(batch_size)

#         mixed_x = lmbda * self.x + (1 - lmbda) * self.x[index, :]
#         y_a, y_b = self.y, self.y[index]
#         return mixed_x, y_a, y_b, lmbda   

def get_train_transforms():
    return albumentations.Compose(
        [
#             albumentations.Resize(params['size'],params['size']),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightness(limit=0.6, p=0.5),
            albumentations.Cutout(
                num_holes=10, max_h_size=12, max_w_size=12,
                fill_value=0, always_apply=False, p=0.5
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
#            albumentations.OneOf([
#                GridMask(num_grid=3, mode=0, rotate=15),
#                GridMask(num_grid=3, mode=2, rotate=15),
#                                ], p=0.7),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(params['size'],params['size']),
            ToTensorV2(p=1.0)
        ]
    )

def get_test_transforms():
        return albumentations.Compose(
            [
                albumentations.Resize(params['size'],params['size']),
                ToTensorV2(p=1.0)
            ]
        )

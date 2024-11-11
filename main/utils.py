import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from pytorch_grad_cam import GradCAM
from skimage.util import view_as_windows
import os

# 定义获取显著区域坐标的函数
def get_significant_areas_coords(model, images, n=8):
    """
    为给定的模型和图像批次计算显著区域的坐标。
    
    参数:
        model (torch.nn.Module): 预训练的模型。
        images (torch.Tensor): 输入图像批次，应该是归一化后的Tensor。
        n (int): 显著区域正方形的边长。
    
    返回:
        list: 每张图像显著区域左上角坐标的列表。

    """
     # 检查输入图像是否含有无效值
    if torch.isnan(images).any() or torch.isinf(images).any():
        raise ValueError("输入图像包含NaN或Inf")
    
    target_layers = [model.layer4[-1]]  # 选择最后的卷积层
    cam = GradCAM(model=model, target_layers=target_layers)
    # 计算每张图片的Grad-CAM显著性图
    try:
        grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    except RuntimeError as e:
        raise RuntimeError(f"Grad-CAM 计算过程中出错: {e}")
    # grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    coords_list = []

    for idx, grayscale_cam in enumerate(grayscale_cams):
        # 检查显著性图是否含有无效值
        if np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
            raise ValueError(f"显著性图包含NaN或Inf，图像索引: {idx}")
        window_shape = (n, n)
        windows = view_as_windows(grayscale_cam, window_shape)
        window_sums = windows.sum(axis=(2, 3))
        max_window_coords = np.unravel_index(window_sums.argmax(), window_sums.shape)
        coords_list.append((max_window_coords[0], max_window_coords[1]))
    
    return coords_list


# 定义获取显著区域坐标的函数
def get_unsignificant_areas_coords(model, images, n=8):
    """
    为给定的模型和图像批次计算显著区域的坐标。
    
    参数:
        model (torch.nn.Module): 预训练的模型。
        images (torch.Tensor): 输入图像批次，应该是归一化后的Tensor。
        n (int): 显著区域正方形的边长。
    
    返回:
        list: 每张图像显著区域左上角坐标的列表。
    """
    target_layers = [model.layer4[-1]]  # 选择最后的卷积层
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    coords_list = []

    for idx, grayscale_cam in enumerate(grayscale_cams):
        window_shape = (n, n)
        windows = view_as_windows(grayscale_cam, window_shape)
        window_sums = windows.sum(axis=(2, 3))
        max_window_coords = np.unravel_index(window_sums.argmin(), window_sums.shape)
        coords_list.append((max_window_coords[0], max_window_coords[1]))
    
    return coords_list
# 主程序
if __name__ == '__main__':
    os.chdir('/data/wzp/project/A3FL')
    from models.resnet import ResNet18

    # 加载预训练的模型
    model = ResNet18()
    model.load_state_dict(torch.load('/data/wzp/project/A3FL/saved/benign_new/cifar10_1900_avg.pt')['model'])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 数据准备
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    # 获取一个批次的图像
    images, _ = next(iter(data_loader))
    if torch.cuda.is_available():
        images = images.cuda()

    # 调用函数，获取显著区域坐标
    coords = get_significant_areas_coords(model, images)
    print(coords)

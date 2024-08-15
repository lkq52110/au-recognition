
"""
Get the grad-cam,
see https://github.com/jacobgil/pytorch-grad-cam
"""

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from model import MACG
from utils import *


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


if __name__ == '__main__':
    image = Image.open('demo_imgs/disfa8.png')
    image = np.array(image)
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    net = MEFARG(num_classes=8, backbone='resnet50', neighbor_num=3, metric='dots')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(torch.load('checkpoints/best_model_fold3.pth', map_location=device))
    net = load_state_dict(net, 'checkpoints/best_model_fold3.pth')

    net = net.eval()

    if torch.cuda.is_available():
        net = net.cuda()
        input_tensor = input_tensor.cuda()

    # 获取模型输出
    output = net(input_tensor)
    output = output.squeeze()  # 移除不必要的批次维度

    # 计算概率并选取最高得分的类别
    probabilities = torch.nn.functional.softmax(output, dim=0)
    top_category = probabilities.argmax().item()

    # 定义Grad-CAM使用的目标层和类别
    target_layers = [net.head.msca]  # 确保此为正确的目标层
    targets = [ClassifierOutputTarget(top_category)]

    # 使用Grad-CAM
    with GradCAM(model=net, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        result_image = Image.fromarray(cam_image)
        result_image.save('disfa8_msca(1).jpg')

    print("Grad-CAM image saved as 'grad.jpg'")

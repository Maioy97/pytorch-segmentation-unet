import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from diceloss import DiceLoss
import os
import data_loaders


def infer(model, data_loader, dest):
    dice_loss = DiceLoss()
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    totalTestLoss = 0

    outputs = []
    # plt.figure(figsize=(16, 4))
    for i, data in enumerate(data_loader):
        image, mask_label = data
        if torch.cuda.is_available():
            (x, y) = (image.to('cuda'), mask_label.to('cuda'))
        else:
            x, y = image, mask_label

        with torch.no_grad():
            pred = model(x)
            totalTestLoss += dice_loss(pred, y)
            outputs.append(pred)
            plt.imsave(os.path.join(dest, f"{i}_mask.png"), image)
            masked_image = np.ma.masked_array(image, pred)
            plt.imsave(os.path.join(dest, f"{i}_masked.png"), masked_image)


if __name__ == '__main__':
    model_path = 'unet_axial_Dataset_best.pt'
    model = torch.load(model_path)
    dataset_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    axial_test_loader = data_loaders.axial_dataset_test(dataset_transforms, 0.2)
    dest_path = 'axial_results_axial_model'
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    infer(model, axial_test_loader, dest_path)



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
    totalTestLoss = 0

    for i, data in enumerate(data_loader):
        image, mask_label = data
        if torch.cuda.is_available():
            (x, y) = (image.to('cuda:0'), mask_label.to('cuda:0'))
        else:
            x, y = image, mask_label

        with torch.no_grad():
            pred = model(x)
            totalTestLoss += dice_loss(pred, y)
            image = image.cpu().detach().numpy()
            mask = pred.cpu().detach().numpy()
            plt.imsave(os.path.join(dest, f"{i}_mask.png"), image, cmap='bone')
            masked_image = np.ma.masked_array(image, mask)
            plt.imsave(os.path.join(dest, f"{i}_masked.png"), masked_image)


if __name__ == '__main__':
    model_path = 'unet_axial_Dataset_best.pt'
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=1, init_features=32, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    dataset_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    axial_test = data_loaders.axial_dataset_test(dataset_transforms)
    axial_test_loader = torch.utils.data.DataLoader(axial_test, shuffle=True,
                                                    batch_size=1)
    dest_path = 'axial_results_axial_model'
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    infer(model, axial_test_loader, dest_path)



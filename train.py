import torch
import numpy as np


import copy
import time
# from logger import Logger
from diceloss import DiceLoss
# loss
# https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
from tqdm import tqdm
from torchvision import transforms
import data_loaders as DataLoaders
import matplotlib.pyplot as plt


def train_model(model, optimizer, data_loaders, num_epochs=25, batch_size=1, save_path="unet_axial_Dataset_best.pt"):
    since = time.time()
    vis_freq = 2
    vis_images = 1000

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dice_loss = DiceLoss()

    history = {'train_loss': [], 'val_loss': []}

    # logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    # best_validation_dsc = 0
    best_valid_loss = 3  # some relativly big loss
    step = 0
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            validation_pred = []
            validation_true = []

            # Iterate over data.
            for i, data in enumerate(data_loaders[phase]):
                # add batches and batch size
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    step += 1
                    y_pred = model(inputs)
                    loss = dice_loss(y_pred, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        history['train_loss'].append(loss.item())
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                    if phase == "val":
                        history['val_loss'].append(loss.item())
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

                        y_true_np = labels.detach().cpu().numpy()
                        validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

                        if (epoch % vis_freq == 0) or (epoch == num_epochs - 1):
                            if i * batch_size < vis_images:
                                tag = "image/{}".format(i)
                                num_images = vis_images - i * batch_size
                                # logger.image_list_summary(
                                #     tag,
                                #     log_images(x, y_true, y_pred)[:num_images],
                                #     step,
                                # )

            if phase == "train" and (step + 1) % 10 == 0:
                # log_loss_summary(logger, loss_train, step)
                loss_train = []
                # scheduler.step()

            if phase == 'val':
                curr_valid_loss = loss_valid[-1]
                if curr_valid_loss < best_valid_loss:
                    best_valid_loss = curr_valid_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, save_path)
                    # print('saved at epoch ', epoch)

                print("Best validation loss: {:4f}".format(best_valid_loss))
            # add visualisation

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return history


def draw_graph(history, plt_path: str):
    plot_path = 'train_val_graph.png'
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plt_path)


if __name__ == '__main__':
    # original model had 3 channels for input
    # https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                            in_channels=1, out_channels=1, init_features=32, pretrained=False)

    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')
    model.to(device)
    if next(model.parameters()).is_cuda:
        print("model running on gpu")
    else:
        print("model running on cpu")

    # get loaders
    BATCH_SIZE = 16
    num_epochs = 32
    dataset_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    axial_dataset_train, axial_dataset_val = DataLoaders.axial_dataset_train(dataset_transforms, 0.2)
    # full_dtaset_test = full_dataset_test(dataset_transforms)

    full_train_loader = torch.utils.data.DataLoader(axial_dataset_train, shuffle=True, batch_size=BATCH_SIZE)
    full_val_loader = torch.utils.data.DataLoader(axial_dataset_val, shuffle=True, batch_size=BATCH_SIZE)
    # full_train_dataset_loader = torch.utils.data.DataLoader(full_dataset_train, shuffle=true, batch_size=BATCH_SIZE)

    data_loaders = {'train': full_train_loader, 'val': full_val_loader}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = train_model(model, optimizer, None, data_loaders, num_epochs, batch_size=BATCH_SIZE)
    plot_path = './axial_train_val.png'
    draw_graph(history, plot_path)

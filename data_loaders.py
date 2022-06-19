# import torch.segmentation_models_pytorch as smp
# single model for all views exp
# experiment with 3 separate models
import gzip
import torch
import glob
import numpy as np
from math import ceil


class JawsDataset(torch.utils.data.Dataset):
    def __init__(self, dicom_file_list, transforms):
        self.dicom_file_list = dicom_file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.dicom_file_list)

    def __getitem__(self, idx):
        dicom_path = self.dicom_file_list[idx]
        label_path = dicom_path.replace('.dicom.npy.gz', '.label.npy.gz')
        dicom_file = gzip.GzipFile(dicom_path, 'rb')
        dicom = np.load(dicom_file)
        label_file = gzip.GzipFile(label_path, 'rb')
        label = np.load(label_file)
        return self.transforms(dicom), self.transforms(label)


def axial_dataset_train(transforms, validation_ratio = 0.1):
    files = glob.glob('dataset/axial/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], transforms),
            JawsDataset(files[:validation_files_count], transforms))


def coronal_dataset_train(transforms, validation_ratio=0.1):
    files = glob.glob('dataset/coronal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], transforms),
            JawsDataset(files[:validation_files_count], transforms))


def sagittal_dataset_train(transforms, validation_ratio=0.1):
    files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], transforms),
            JawsDataset(files[:validation_files_count], transforms))


def axial_dataset_test(transforms):
    files = glob.glob('dataset/axial/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, transforms)


def coronal_dataset_test(transforms):
    files = glob.glob('dataset/coronal/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, transforms)


def sagittal_dataset_test(transforms):
    files = glob.glob('dataset/sagittal/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, transforms)


def full_dataset_train(transforms, validation_ration=0.1):
    sagital_files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    files = sagital_files
    axial_files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    files.extend(axial_files)
    coronal_files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    files.extend(coronal_files)
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ration)

    return (JawsDataset(files[validation_files_count:], transforms),
            JawsDataset(files[:validation_files_count], transforms))


def full_dataset_test(transforms):
    sagital_files = glob.glob('dataset/sagittal/test/**/*.dicom.npy.gz')
    files = sagital_files
    axial_files = glob.glob('dataset/sagittal/test/**/*.dicom.npy.gz')
    files.extend(axial_files)
    coronal_files = glob.glob('dataset/sagittal/test/**/*.dicom.npy.gz')
    files.extend(coronal_files)
    assert len(files) > 0

    return JawsDataset(files, transforms)
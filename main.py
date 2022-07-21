import SimpleITK as sitk
import matplotlib.pyplot as plt
import radiomics
import numpy as np
from SimpleITK import Image


def show_dicom(image: Image, mask_image: Image, slice: int):
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(image)[slice, :, :], cmap="gray")
    plt.axis('off')
    plt.title("Печень")
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(mask_image)[slice, :, :])
    plt.title("Маска")
    plt.axis('off')
    plt.show()


def calculate_firstorder_parameters(image: Image, mask_image: Image, firstorder_features: list[str]):
    param = radiomics.firstorder.RadiomicsFirstOrder(image, mask_image)
    result = {}
    for feature in firstorder_features:
        param.enableFeatureByName(feature, True)
        result.update(param.execute())
    return result


def calculate_shape_parameters(image: Image, mask_image: Image, shape_features: list[str]):
    param = radiomics.shape.RadiomicsShape(image, mask_image)
    result = {}
    for feature in shape_features:
        param.enableFeatureByName(feature, True)
        result.update(param.execute())
    return result


def print_features(features: dict):
    for (key, val) in features.items():
        print('  ', key, ':', val)


def calculate_features_without_pyradiomics(image: Image, mask_image: Image):
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    pixel_spacing = image.GetSpacing()
    index_of_pixel_array = []
    pixel_array = []
    col_min = len(mask_array[0][0])
    col_max = 0
    row_min = len(mask_array[0])
    row_max = 0
    for slice in range(mask_array.shape[0]):
        for row in range(mask_array.shape[1]):
            for col in range(mask_array.shape[2]):
                if mask_array[slice, row, col] != 0:
                    index_of_pixel_array.append([slice, row, col])
                    if col_min > col:
                        col_min = col
                    if col_max < col:
                        col_max = col
                    if row_min > row:
                        row_min = row
                    if row_max < row:
                        row_max = row
    coordinate_col = (col_max - col_min) * pixel_spacing[0]
    coordinate_row = (row_max - row_min) * pixel_spacing[1]
    coordinate_slice = (index_of_pixel_array[-1][0] - index_of_pixel_array[0][0]) * pixel_spacing[2]
    for index_pixel in index_of_pixel_array:
        pixel_array.append(image_array[index_pixel[0]][index_pixel[1]][index_pixel[2]])
    volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2] * len(pixel_array)
    return {'Mean': np.mean(pixel_array), 'Standard Deviation': np.std(pixel_array), 'Median': np.median(pixel_array),
            'ColumnSize': coordinate_col, 'RowSize': coordinate_row, 'SliceSize': coordinate_slice, 'Volume': volume}


if __name__ == "__main__":
    image_path: str = "data/medical_image.nrrd"
    mask_path: str = "data/mask.nrrd"
    image: Image = sitk.ReadImage(image_path)
    mask_image: Image = sitk.ReadImage(mask_path)
    slice: int = 180
    show_dicom(image, mask_image, slice)

    firstorder_features: list[str] = ['Mean', 'StandardDeviation', 'Median']
    print('\nCalculated first order features with pyradiomics: ')
    print_features(calculate_firstorder_parameters(image, mask_image, firstorder_features))
    shape_features: list[str] = ['Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'Maximum2DDiameterSlice',
                                 'MinorAxisLength',
                                 'MajorAxisLength', 'MeshVolume', 'VoxelVolume']
    print('\nCalculated shape features with pyradiomics: ')
    print_features(calculate_shape_parameters(image, mask_image, shape_features))
    print('\nCalculated features without pyradiomics: ')
    print_features(calculate_features_without_pyradiomics(image, mask_image))

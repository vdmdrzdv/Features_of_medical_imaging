import SimpleITK as sitk
import matplotlib.pyplot as plt
import radiomics


def show_dicom(image_path: str, mask_path: str, slice: int):
    image = sitk.ReadImage(image_path)
    mask_image = sitk.ReadImage(mask_path)
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


def calculate_firstorder_parameters(image_path: str, mask_path: str, firstorder_features: list[str]):
    image = sitk.ReadImage(image_path)
    mask_image = sitk.ReadImage(mask_path)
    param = radiomics.firstorder.RadiomicsFirstOrder(image, mask_image)
    result = {}
    for feature in firstorder_features:
        param.enableFeatureByName(feature, True)
        result.update(param.execute())
    return result


def calculate_shape_parameters(image_path: str, mask_path: str, shape_features: list[str]):
    image = sitk.ReadImage(image_path)
    mask_image = sitk.ReadImage(mask_path)
    param = radiomics.shape.RadiomicsShape(image, mask_image)
    result = {}
    for feature in shape_features:
        param.enableFeatureByName(feature, True)
        result.update(param.execute())
    return result


def print_features(features: dict):
    for (key, val) in features.items():
        print('  ', key, ':', val)


if __name__ == "__main__":
    image_path: str = "data/medical_image.nrrd"
    mask_path: str = "data/mask.nrrd"
    slice: int = 174
    show_dicom(image_path, mask_path, slice)

    firstorder_features: list[str] = ['Mean', 'StandardDeviation', 'Median']
    print('Calculated first order features: ')
    print_features(calculate_firstorder_parameters(image_path, mask_path, firstorder_features))

    shape_features: list[str] = ['Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'Maximum2DDiameterSlice',
                                 'MinorAxisLength',
                                 'MajorAxisLength', 'MeshVolume', 'VoxelVolume']
    print('Calculated shape features: ')
    print_features(calculate_shape_parameters(image_path, mask_path, shape_features))

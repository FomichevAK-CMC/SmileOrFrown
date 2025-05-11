import os
import shutil, random
from PIL import Image

from pathlib import Path


# Iterate over all files in the directory
def merge_prepare_datasets(set_list, crop_to, to_dir):
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    ind = 0
    for set in set_list:
        filenames = os.listdir(set)
        for filename in filenames:
            # Construct new file name with prefix
            name, ext = filename.split('.')
            new_filename = str(ind) + '.' + ext
            ind += 1
            # Construct new file path
            new_file_path = os.path.join(to_dir, new_filename)
            old_file_path = os.path.join(set, filename)
            # Rename the file
            crop_center(old_file_path, new_file_path, crop_to)
            print(f'Copied and renamed: {filename} -> {new_filename}')


def crop_center(image_path, output_path, crop_size):
    # Open the image
    with Image.open(image_path) as img:
        # Get the original image dimensions
        width, height = img.size

        # Calculate the coordinates for cropping
        left = (width - crop_size[0]) // 2
        top = (height - crop_size[1]) // 2
        right = (width + crop_size[0]) // 2
        bottom = (height + crop_size[1]) // 2

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_img.save(output_path)

def save_image_with_postfix(image, postfix, img_name, to_dir = None):
    img_name = Path(img_name)
    if to_dir is None:
        to_dir = img_name.parent
        print("parent:", img_name, to_dir)
    image.save(Path(to_dir).joinpath(img_name.stem + postfix + img_name.suffix))


def shift_image(image_path, shift, fill_color, to_dir=None):
    if shift[0] == 0 and shift[1] == 0:
        return
    image = Image.open(image_path)
    new_image = Image.new("RGB", image.size, fill_color)
    new_image.paste(image, shift)
    save_image_with_postfix(new_image, f"_s-({shift[0]:.2f},{shift[1]:.2f})", image_path, to_dir)


def rotate_image(image_path, angle, fill_color, to_dir=None):
    if angle == 0:
        return
    image = Image.open(image_path).rotate(angle, fillcolor=fill_color)
    save_image_with_postfix(image, f"_r-({angle:.2f})", image_path, to_dir)


def zoom_image(image_path, zoom_factor, fill_color, to_dir=None):
    if zoom_factor == 0:
        return

    image = Image.open(image_path)
    new_image = Image.new("RGB", image.size, fill_color)
    half = image.size[0] // 2
    cent_x = int(zoom_factor * image.size[0]) // 2
    cent_y = int(zoom_factor * image.size[1]) // 2
    box = ((cent_x - half),
           (cent_y - half),
           (cent_x + half),
           (cent_y + half))
    print(box)

    image = image.resize((int(image.size[0] * zoom_factor), int(image.size[1] * zoom_factor)))
    image = image.crop(box)
    new_image.paste(image)
    save_image_with_postfix(new_image, f"_z-({zoom_factor:.2f})", image_path, to_dir)

def dataset_grid_shift(dataset_f, grid_size, step, fill_color=(255, 255, 255), to_dir=None):
    half_size = grid_size / 2 * step
    shift_grid = [[(int(-half_size + step * m), int(-half_size + step * k)) for k in range(grid_size)] for m in range(grid_size)]

    for img_path in os.listdir(dataset_f):
        for r in shift_grid:
            for s in r:
                shift_image(os.path.join(dataset_f, img_path), s, (249, 247, 234))


def dataset_rotations(dataset_f, from_to=(-32, 32), step=5, to_dir=None):
    num = (from_to[1] - from_to[0]) // step + 1
    for img_path in os.listdir(dataset_f):
        for i in range(num):
            rotate_image(os.path.join(dataset_f, img_path), from_to[0] + step * i, (249, 247, 234))




if __name__ == '__main__':
    set_list = ['samples/sset1_split','samples/sset2_split', 'samples/sset3_split']
    to_dir = "samples/sset_full"
    merge_prepare_datasets(set_list,(256, 256), to_dir)
    dataset_rotations(to_dir, (-15, 15), 5)
    dataset_grid_shift(to_dir, 4, 10)

    set_list = ['samples/fset1_split','samples/fset2_split', 'samples/fset3_split']
    to_dir = "samples/fset_full"
    merge_prepare_datasets(set_list,(256, 256), to_dir)
    dataset_rotations(to_dir, (-15, 15), 5)
    dataset_grid_shift(to_dir, 4, 10)
from tensorflow import keras
import shutil
import os

download_path = 'C:/Data/landmarks_dataset/'
extraction_path = download_path + 'datasets/'
destination_path = 'C:/Data/landmarks_dataset/0'
url_base = 'https://s3.amazonaws.com/google-landmark/train/images_'
first_chunk = 0
last_chunk = 499


def download_and_extract(start, end):
    for nr in range(start, end + 1):
        url = url_base + ((3 - len(str(nr))) * '0' + str(nr)) + '.tar'
        keras.utils.get_file('chunk.tar',
                             url,
                             cache_dir=download_path,
                             extract=True)
        os.remove(extraction_path + 'chunk.tar')


def move_all_images(path_in, path_out):
    for content in os.walk(path_in, topdown=False):
        if len(content[2]) != 0:
            for file in os.listdir(content[0]):
                shutil.move(os.path.join(content[0], file), os.path.join(path_out, file))


def clean_up_folder(directory):
    while len(os.listdir(directory)) != 0:
        for content in os.walk(directory):
            if len(os.listdir(content[0])) == 0:
                os.rmdir(content[0])


def fetch_images():
    download_and_extract(first_chunk, last_chunk)
    move_all_images(extraction_path, destination_path)
    clean_up_folder(extraction_path)

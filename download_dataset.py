import os
from torchvision.datasets.utils import download_and_extract_archive

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), './autodl-tmp/data-imagenetv2/')
URL = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'

if __name__ == '__main__':
    # the file is named .tar.gz but is an uncompressed tar: saved as .tar so it is extracted as one
    download_and_extract_archive(URL, download_root=DATA_ROOT, filename='imagenetv2-matched-frequency.tar')
    print(f'[+] ImageNetV2 dataset ready at {DATA_ROOT}')

---
layout: page
title: "[pytorch] 변환(Transform)을 활용한 이미지 증강(Image Augmentation) 적용"
description: "[pytorch] 변환(Transform)을 활용한 Image Augmentation 적용 방법에 대해 알아보겠습니다."
headline: "[pytorch] 변환(Transform)을 활용한 Image Augmentation 적용 방법에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, pytorch, image transform, image augmentation, 이미지 증강, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-02
---

**데이터 증강(Data Augmentation)**은 학습을 위한 데이터에 변형을 가하여 데이터의 규모를 키울 뿐만 아니라, 데이터에 대한 변형된 다양한 케이스를 학습하게 만들 수 있는 좋은 수단 중 하나입니다. 또한, 모델이 과적합(overfitting) 되는 것을 방지해주는 효과도 있습니다. 어떨때는 모델을 변경하는 것보다 이미지 증강에 더 신경을 써야하는 경우도 있습니다. 모델이 앞으로 학습할 데이터셋에 어떠한 전처리를 적용하느냐에 따라서 성능이 달라집니다.

이는 논문에서도 흔히 확인할 수 있는 대목이기도 하며, 따라서 SOTA 모델에 주입한 다양한 Data Augmentation 방법론과 그에 따른 실험 결과가 명시되어있는 것도 어렵지 않게 확인할 수 있습니다.

이미지 데이터셋에 증강을 적용하는 방법은 다양하게 존재합니다. 예를 들면, 다음과 같습니다.

- Flip (Horizontal, Vertical)
- Random Crop
- Shear
- Rotate
- Zoom
- Blur

이번 튜토리얼에서는 **PyTorch의 변환기(Transform)**를 활용하여 손쉽게 **이미지 데이터셋 증강**을 적용하는 방법에 대해 알아보겠습니다.

## 이미지 증강(Image Augmentation) 적용하기

### 샘플 데이터셋 다운로드

- 본 튜토리얼 에서는 kaggle에 공개된 `cats and dogs` 데이터셋을 활용합니다.

```python
import urllib.request
import zipfile
import glob
import os
from PIL import Image, UnidentifiedImageError, ImageFile


# 이미지 Validation을 수행하고 Validate 여부를 return 합니다.
def validate_image(filepath):
    # image extensions
    image_extensions = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp','.png', '.avif', '.gif']
    # 이미지 파일 확장자를 가진 경우
    if any(filepath.endswith(s) for s in image_extensions):   
        try:
            # PIL.Image로 이미지 데이터를 로드하려고 시도합니다.
            img = Image.open(filepath).convert('RGB')
            img.load()
        except UnidentifiedImageError: # corrupt 된 이미지는 해당 에러를 출력합니다.
            print(f'Corrupted Image is found at: {filepath}')
            return False
        except (IOError, OSError): # Truncated (잘린) 이미지에 대한 에러를 출력합니다.
            print(f'Truncated Image is found at: {filepath}')
            return False
        else:
            return True
    else:
        print(f'File is not an image: {filepath}')
        return False

def download_dataset(download_url, folder, default_folder='tmp'):
    # 데이터셋을 다운로드 합니다.
    urllib.request.urlretrieve(download_url, 'datasets.zip')

    # 다운로드 후 tmp 폴더에 압축을 해제 합니다.
    local_zip = 'datasets.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(f'{default_folder}/')
    zip_ref.close()

    # 잘린 이미지 Load 시 경고 출력 안함
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # image 데이터셋 root 폴더
    root = f'{default_folder}/{folder}' 

    dirs = os.listdir(root)

    for dir_ in dirs:
        folder_path = os.path.join(root, dir_)
        files = os.listdir(folder_path)

        images = [os.path.join(folder_path, f) for f in files]
        for img in images:
            valid = validate_image(img)
            if not valid:
                # corrupted 된 이미지 제거
                os.remove(img)

    folders = glob.glob(f'{default_folder}/{folder}/*')
    print(folders)
    return folders
```

> 개와 고양이 데이터셋 다운로드

```python
# 개, 고양이 데이터셋 다운로드를 받습니다.
# 받은 데이터는 tmp/PetImages 폴더 안에 저장됩니다.
# 중간에 Truncated 된 이미지가 존재합니다.
download_dataset('https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip', 'PetImages')
```



## 학습에 활용할 데이터셋 준비

```python
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 랜덤 시드 설정
torch.manual_seed(321)

# 이미지 크기를 224 x 224 로 조정합니다
IMAGE_SIZE = 224
```

`torchvision.datasets.ImageFolder`

- `root`: 이미지 폴더가 위치한 루트 폴더의 경로를 입력합니다.
- `transform`: 이미지 변형에 대한 함수를 입력합니다.
- `traget_transform`: `target` 값에 대한 변형하는 함수를 입력합니다. 원핫인코딩을 적용할 수 있습니다.
- `loader`: 이미지 로더 함수를 적용할 수 있습니다.
- `is_valid_file`: 이미지의 corrupt여부를 체크할 수 있습니다.

```python
# 이미지 폴더로부터 데이터를 로드합니다.
original_dataset = ImageFolder(root='tmp/PetImages',                            # 다운로드 받은 폴더의 root 경로를 지정합니다.
                               transform=transforms.Compose([                   # Resize 후 정규화(0~1)를 수행합니다.
                                   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
                                   transforms.ToTensor()
                               ]))
```

```python
# 데이터 로더를 생성합니다.
original_loader = DataLoader(original_dataset, # 이전에 생성한 original_dataset를 로드 합니다.
                             batch_size=16,    # 배치사이즈
                             shuffle=False,     # 셔플 여부
                             num_workers=1 
                            )
```

```python
# 1개의 배치를 추출합니다.
original_images, labels = next(iter(original_loader))

# 이미지의 shape을 확인합니다. 224 X 224 RGB 이미지 임을 확인합니다. 
# (batch_size, channel, height, width)
original_images.shape
```

```
torch.Size([16, 3, 224, 224])
```

> 단일 이미지 시각화

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
# permute로 이미지의 shape를 다음과 같이 변경합니다
# (height, width, channel)
plt.imshow(original_images[0].permute(1, 2, 0))
plt.grid(False)
plt.axis('off')
plt.show()
```

![01](../images/2023-01-02/01.png)

개와 고양이 이미지 데이터셋의 원본 사진을 시각화 합니다. 처음 3장의 이미지만 출력합니다.

```python
fig, axes = plt.subplots(3, 1)
fig.set_size_inches(4, 6)

for idx in range(3):
    axes[idx].imshow(original_images[idx].permute(1, 2, 0))
    axes[idx].set_axis_off()
fig.tight_layout()
plt.show()
```

![02](../images/2023-01-02/02.png)



이미지 변환기(Image Transforms)는 [공식도큐먼트 링크](https://pytorch.org/vision/stable/transforms.html)에서 세부 함수의 인자(parameter)와 사용 예시를 확인할 수 있습니다. 아래는 몇 가지 함수를 적용해보고 원본 이미지와 비교하여 시각화한 결과입니다.

> transform 적용

```python
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),                    # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 중앙 Crop
    transforms.ToTensor()
])

# 이미지 폴더로부터 데이터를 로드합니다.
transform_dataset = ImageFolder(root='tmp/PetImages',                  # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                transform=image_transform)

# 데이터 로더를 생성합니다.
transform_loader = DataLoader(transform_dataset, # 이전에 생성한 transform_dataset를 적용합니다.
                              batch_size=16,     # 배치사이즈
                              shuffle=False,      # 셔플 여부
                              num_workers=1 
                             )
```

1개의 배치만 가져옵니다.

```python
transform_images, labels = next(iter(transform_loader))
```

> 원본, 변형본의 이미지 비교

```python
fig, axes = plt.subplots(3, 2)
fig.set_size_inches(4, 6)

for idx in range(3):
    axes[idx, 0].imshow(original_images[idx].permute(1, 2, 0))
    axes[idx, 0].set_axis_off()
    axes[idx, 0].set_title('Original')
    axes[idx, 1].imshow(transform_images[idx].permute(1, 2, 0))
    axes[idx, 1].set_axis_off()
    axes[idx, 1].set_title('CenterCrop')
fig.tight_layout()
plt.show()
```

![03](../images/2023-01-02/03.png)



이미지 변형을 적용하고 시각화 하는 부분을 함수 형태로 만듭니다.

```python
def create_loader(transform):
    # 이미지 폴더로부터 데이터를 로드합니다.
    transform_dataset = ImageFolder(root='tmp/PetImages', # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                    transform=transform)

    # 데이터 로더를 생성합니다.
    transform_loader = DataLoader(transform_dataset,      # 이전에 생성한 transform_dataset를 적용합니다.
                                  batch_size=16,          # 배치사이즈
                                  shuffle=False,          # 셔플 여부
                                  num_workers=1 
                                 )
    
    transform_images, labels = next(iter(transform_loader))
    
    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(4, 6)

    for idx in range(3):
        axes[idx, 0].imshow(original_images[idx].permute(1, 2, 0))
        axes[idx, 0].set_axis_off()
        axes[idx, 0].set_title('Original')
        axes[idx, 1].imshow(transform_images[idx].permute(1, 2, 0))
        axes[idx, 1].set_axis_off()
        axes[idx, 1].set_title('Transformed')
        
    fig.tight_layout()
    plt.show()
```

## 변형(Transform) 적용

-  [공식도큐먼트 링크](https://pytorch.org/vision/stable/transforms.html)

`ColorJitter`

- `brightness`: 밝기 조절 (0~1)
- `contrast`: 대비 조절 (0~1)
- `saturation`: 채도 조절 (0~1)
- `hue`: 휴 조절 (-0.5~0.5)

```python
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),              
    # ColorJitter 적용
    transforms.ColorJitter(brightness=(0.5, 0.9), 
                           contrast=(0.4, 0.8), 
                           saturation=(0.7, 0.9),
                           hue=(-0.2, 0.2),
                          ),
    transforms.ToTensor()
])

create_loader(image_transform)
```

![04](../images/2023-01-02/04.png)



`RandomHorizontalFlip(p=0.5)`
- Horizontal Flip의 확률을 조절합니다. 

```python
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),              
    # RandomHorizontalFlip 적용
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.ToTensor()
])

create_loader(image_transform)
```

![05](../images/2023-01-02/05.png)

`GaussianBlur(kernel_size, sigma=(0.1, 2.0))`
- `kernel_size`: 가우시안 커널의 크기
- `sigma`: 커널의 Standard Deviation

```python
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),              
    # GaussianBlur 적용
    transforms.GaussianBlur(kernel_size=(19, 19), sigma=(1.0, 2.0)),
    transforms.ToTensor()
])

create_loader(image_transform)
```

![06](../images/2023-01-02/06.png)








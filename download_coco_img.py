from pycocotools.coco import COCO
import os
import requests
from io import BytesIO
from PIL import Image


def download_and_save_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            print(f"Image saved successfully to {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


dataDir = r'E:\project_3_image_captioning\mPLUG_official_code\AliceMind\mPLUG\data'
dataType = 'val2014'
annFile = '{}/coco_object/annotations/captions_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)


imgIds = coco.getImgIds()
print(coco.loadImgs(imgIds[0]))
anns = coco.loadAnns(coco.getAnnIds(imgIds[0]))
print(anns)
# print(f"Number of images: {len(imgIds)}")
# subset_imgIds = imgIds


# subset_dir = r'E:\project_3_image_captioning\mPLUG_official_code\AliceMind\mPLUG\img_root\coco_2014'
# os.makedirs(subset_dir, exist_ok=True)

# for imgId in subset_imgIds:
#     img = coco.loadImgs(imgId)[0]
#     coco_url = img['coco_url']
#     file_name = img['file_name']
#     save_path = os.path.join(subset_dir, file_name)
#     download_and_save_image(coco_url, save_path)

# print(f"Downloaded {len(subset_imgIds)} images to {subset_dir}")

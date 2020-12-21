from PIL import Image,ImageDraw
import torch
import torchvision
import transforms as T
import os
import numpy as np
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# mask = Image.open('./PennFudanPed/PedMasks/FudanPed00001_mask.png')
# # each mask instance has a different color, from zero to N, where
# # N is the number of instances. In order to make visualization easier,
# # let's adda color palette to the mask.
# mask.putpalette([
#     0, 0, 0, # black background
#     255, 0, 0, # index 1 is red
#     255, 255, 0, # index 2 is yellow
#     255, 153, 0, # index 3 is orange
# ])
# mask.save('./example.png')

def tensor_to_PILImage(tensor):
    unloader=torchvision.transforms.ToPILImage()
    PILImage=tensor.cpu().clone()
    PILImage=PILImage.squeeze(0)
    PILImage=unloader(PILImage)
    return PILImage
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_test=PennFudanDataset('PennFudanPed',get_transform(train=False))
for i in range(len(dataset_test)):
    img,_ = dataset_test[i]
    model = torch.load('\model_mengfan.pkl')
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        bounding_boxs=prediction[0]['boxes']
        scores=prediction[0]['scores']
        img=tensor_to_PILImage(img)
        for bounding_box,score in zip(bounding_boxs,scores):
            draw=ImageDraw.Draw(img)
            if score>0.90:
                draw.rectangle(xy=(int(bounding_box[0]),int(bounding_box[1]),int(bounding_box[2]),int(bounding_box[3])),fill=None,outline='green',width=2)
                draw.text((int(bounding_box[0])+7,int(bounding_box[1])+7),str(round(float(score),4)),fill=(0,255,0))
            else:
                draw.rectangle(xy=(int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])),fill=None, outline='red', width=2)
                draw.text((int(bounding_box[0]) + 7, int(bounding_box[1]) + 7), str(round(float(score), 4)),fill=(255, 0, 0))
        img.save('/Data_HDD/INT408_20/INT408_5/mengfan/INT408_a2/PennFudanPed/Bounding_img/'+'BOX'+str(i+1)+'.png')


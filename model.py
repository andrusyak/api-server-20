import io, json
from PIL import Image

import torch
from torchvision import models, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18(pretrained=True)
resnet18.to(device)
resnet18.eval()


with open("data/imagenet_class_index.json") as f:
	idx_to_class = json.load(f)


def transform_image(img_bytes):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	data_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])

	image = Image.open(io.BytesIO(img_bytes))
	return data_transforms(image).unsqueeze(0)


def get_prediction(img_bytes):
	tensor = transform_image(img_bytes).to(device)

	with torch.no_grad():
		outputs = resnet18(tensor)
	_, pred_idx = outputs.max(1)

	return idx_to_class[str(pred_idx.item())]


# def batch_prediction(img_bytes_batch):
# 	tensors = [transform_image(img_bytes) for img_bytes in img_bytes_batch]
# 	tensor = torch.cat(tensors).to(device)

# 	with torch.no_grad():
# 		outputs = resnet18(tensor)
# 	_, pred_idx = outputs.max(1)
	
# 	return [idx_to_class[str(idx.item())] for idx in pred_idx]

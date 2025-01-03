import torch
from PIL import Image, ImageFile
import csv
from torchvision import transforms
from utils import *
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model

BATCH_SIZE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "liuhaotian/llava-v1.6-vicuna-7b"

from data.baid_dataset import *
# from model.arch import ImageEncoder, LLaVATextEncoder
torch.autograd.set_detect_anomaly(True)
max_len = 512
BATCH_SIZE = 1
learning_rate = 1e-4
epochs = 20
SEED = 34
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose([
    # transforms.Resize((256, 256)), 
    # transforms.RandomCrop(224), 
    # transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = OriginalBBDataset('dataset', type='train')
print(train_dataset.__len__())

file_path = "aesthetics_comment.csv"

def call_llava(prompt, image_file):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    return eval_model(args))

def write_comments(prompt, file_path):
    with open(file_path, mode='a', newline='') as f:
        for i in range(len(train_dataset)):
            image_name, _, _ = train_dataset[i]
            llava_text = call_llava(prompt, image_name)
            llava_text = llava_text[0].replace('\n', '')
            writer = csv.writer(f)
            writer.writerow([llava_text])

write_comments("Describe artistic aesthetics of an image in a sentence", "aesthetics_comment.csv")
write_comments("Desctibe artistic style of an image in a sentence", "style_comment.csv")
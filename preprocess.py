import torch
from torch.utils.data import DataLoader
from PIL import ImageFile
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from torchvision import transforms
from data.baid_dataset import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CSV file
file_path = 'BAID/dataset/train_set.csv'
data = pd.read_csv(file_path)

# Apply Box-Cox transformation to the 'score' column
transformed_scores, lambda_bc = stats.boxcox(data['score'])
mean_transformed = transformed_scores.mean()
std_transformed = transformed_scores.std()
print(lambda_bc)
print(mean_transformed, std_transformed)
standardized_scores = (transformed_scores - mean_transformed) / std_transformed

transformed_data = pd.DataFrame({
    'image': data['image'],
    'score': standardized_scores
})

# Save the transformed DataFrame to a new CSV file
transformed_file_path = 'box_cox_train_set.csv'
transformed_data.to_csv(transformed_file_path, index=False)
plt.rcParams["font.size"] = 20
plt.figure(figsize=(12, 8))
plt.hist(transformed_data['score'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Transformed Scores', fontsize=26)
plt.xlabel('Transformed Score', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.grid(True)
plt.savefig('transformed_data.png')
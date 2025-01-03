import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CLIPLoss(nn.Module):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    def forward(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0
        return loss.mean()

class LITALoss(nn.Module):
    def __init__(self, LAMBDA):
        super(LITALoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.clip_loss = CLIPLoss(1.00)
        self.LAMBDA = LAMBDA
    def forward(self, outputs, annotation, aesthetic_text, style_text, aesthetic_img, style_img):
        mse_loss = self.mse_loss(outputs, annotation)
        aesthetic_distance_loss = self.clip_loss(aesthetic_img, aesthetic_text)
        style_distance_loss = self.clip_loss(style_img, style_text)
        return mse_loss + self.LAMBDA*aesthetic_distance_loss + self.LAMBDA*style_distance_loss

def pre_transform(tensor):
    param_l = -3.2889243394393333
    mean = 0.30090352908318774
    std = 0.0015849083753575694
    box_cox = (torch.power(tensor, param_l) - 1) / param_l
    standarized = (box_cox - mean) / std
    return standarized

def inverse_transform(tensor):
    param_l = -3.2889243394393333
    mean = 0.30090352908318774
    std = 0.0015849083753575694
    inv_standarized = tensor * std + mean
    if param_l != 0:
        original_tensor = torch.pow(param_l*inv_standarized + 1, 1 / param_l)
    else:
        original_tensor = torch.exp(inv_standarized)
    return original_tensor

def calculate_plcc(preds, labels):
    pred_mean = torch.mean(preds)
    label_mean = torch.mean(labels)
    pred_norm = preds - pred_mean
    label_norm = labels - label_mean
    bunbo = torch.sqrt(torch.sum(label_norm*label_norm).double() * torch.sum(pred_norm*pred_norm).double())
    bunshi = torch.sum(label_norm.double() * pred_norm.double())
    return bunshi / bunbo

def calculate_srcc(preds, labels):
    n = preds.size(0)
    
    # 予測値とラベルのランクを計算
    preds_rank = preds.argsort().argsort()
    labels_rank = labels.argsort().argsort()
    
    # ランクの差を計算
    rank_diff = preds_rank.float() - labels_rank.float()
    
    # スピアマンの順位相関係数を計算
    rank_diff_sq = rank_diff.pow(2)
    spearman_corr = 1 - (6 * rank_diff_sq.sum()) / (n * (n**2 - 1))
    
    return spearman_corr

def aesthetic_score(tensor, batch_size):
    assert tensor.shape[0] == batch_size
    score_list = []
    for i in range(batch_size):
        score_distribution = tensor[i]
        average_score = 0
        for j in range(10):
            average_score += (j+1)*score_distribution[j]
        score_list.append(average_score.item())
    return torch.FloatTensor(score_list)


def calculate_mask(im, attention):
    att_mat = torch.stack(attention).squeeze(1).cpu()
    att_mat = torch.mean(att_mat, dim=1)

    # 残差接続を考慮
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # 重み行列の再帰的な乗算
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # 出力トークンから入力空間への注意を取得
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis] 
    return mask

def save_attention_map(image_path, original_attention, style_attention, aesthetic_attention, original_score=None, ours_score=None, label_score=None):
    number = image_path.split('/')[-1].split('.')[0]
    im = Image.open(image_path)
    original = calculate_mask(im, original_attention)
    style = calculate_mask(im, style_attention)
    aesthetics = calculate_mask(im, aesthetic_attention)
    

    # 注意マップの表示
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 4))
    fig.suptitle(f'Baseline: {original_score}, Ours: {ours_score}, Ground-Truth: {label_score}')
    ax1.set_title('Original')
    ax2.set_title('Baseline Attention Map')
    ax3.set_title('Style Attention Map')
    ax4.set_title('Aesthetics Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(original)
    _ = ax3.imshow(style)
    _ = ax4.imshow(aesthetics)
    plt.show()
    plt.savefig(f'visualization_84/map_{number}.pdf')
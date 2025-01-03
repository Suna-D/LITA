import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import argparse
from tqdm import tqdm
from PIL import ImageFile
from utils import *
from data.baid_dataset import *
from model.arch import *

torch.autograd.set_detect_anomaly(True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BOXCOXMAX = 1.8871280425787964
BOXCOXMIN = -5.32738830731476

def train(args, text_encoder, image_encoder, trial=None):
    criterion = LITALoss(0.35)
    text_encoder.eval()
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10, f'experiment_num...{args.experiment_num}')
        image_encoder.train()
        total_loss = 0
        score_list = torch.tensor([]).to(device)
        output_list = torch.tensor([]).to(device)
        for image_path, image, annotation, aesthetics_llava, style_llava in tqdm(train_loader):
            image = image.float().to(device)
            annotation = annotation.view(-1, 1).float().to(device)
            optimizer.zero_grad()

            aesthetics_llava = bert_tokenizer(aesthetics_llava, padding=True, truncation=True, return_tensors='pt')
            aesthetics_input_ids = aesthetics_llava['input_ids'].to(device)
            aesthetics_attention_mask = aesthetics_llava['attention_mask'].to(device)
            aesthetics_llava_embed = text_encoder(input_ids=aesthetics_input_ids, attention_mask=aesthetics_attention_mask).last_hidden_state[:, 0, :]
            style_llava = bert_tokenizer(style_llava, padding=True, truncation=True, return_tensors='pt')
            style_input_ids = style_llava['input_ids'].to(device)
            style_attention_mask = style_llava['attention_mask'].to(device)
            style_llava_embed = text_encoder(input_ids=style_input_ids, attention_mask=style_attention_mask).last_hidden_state[:, 0, :]

            outputs, aesthetics_embed, style_embed = image_encoder(image)
            loss = criterion(outputs, annotation, aesthetics_llava_embed.to(device), style_llava_embed.to(device), aesthetics_embed.to(device), style_embed.to(device))
            loss.backward()          
            optimizer.step()
            total_loss += loss.item()
            annotation = inverse_transform(annotation)
            outputs = torch.clamp(outputs, max=BOXCOXMAX, min=BOXCOXMIN)
            outputs = inverse_transform(outputs)
            score_list = torch.cat((score_list, annotation.squeeze().to(device)))
            output_list = torch.cat((output_list, outputs.squeeze().to(device)))

        image_encoder.eval()
        valid_score_list = torch.tensor([]).to(device)
        valid_output_list = torch.tensor([]).to(device)
        with torch.inference_mode():
            for _, image, annotation in tqdm(valid_loader):
                image = image.float().to(device)
                annotation = annotation.view(-1, 1).float().to(device)
                val_outputs, _ , _= image_encoder(image)
                val_outputs = torch.clamp(val_outputs, max=BOXCOXMAX, min=BOXCOXMIN)
                val_outputs = inverse_transform(val_outputs)
                valid_score_list = torch.cat((valid_score_list, annotation.squeeze()))
                valid_output_list = torch.cat((valid_output_list, val_outputs.squeeze()))
        average_loss = total_loss / len(train_loader)
        epoch_plcc = calculate_plcc(score_list, output_list).item()
        epoch_srcc = calculate_srcc(score_list, output_list).item()
        epoch_val_plcc = calculate_plcc(valid_score_list, valid_output_list).item()
        epoch_val_srcc = calculate_srcc(valid_score_list, valid_output_list).item()
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {average_loss}, Train PLCC: {epoch_plcc}, Train SRCC: {epoch_srcc}, Val PLCC: {epoch_val_plcc} Val SRCC: {epoch_val_srcc}')
    torch.save(image_encoder.state_dict(), f'checkpoint/baid_model_{str(args.experiment_num)}.pth')


def test(args, trained_model):
    trained_model.to(device)
    trained_model.eval()
    test_score_list = torch.tensor([]).to(device)
    test_output_list = torch.tensor([]).to(device)
    path_list = []
    with torch.inference_mode():
        for image_path, image, annotation in tqdm(test_loader):
            annotation = annotation.view(-1, 1).float().to(device)
            image = image.float().to(device)
            outputs, _, _ = trained_model(image)
            outputs = torch.clamp(outputs, max=BOXCOXMAX, min=BOXCOXMIN)
            outputs = inverse_transform(outputs)
            outputs = torch.clamp(outputs, max=10.0, min=0)
            test_score_list = torch.cat((test_score_list, annotation.squeeze()))
            test_output_list = torch.cat((test_output_list, outputs.squeeze()))
            path_list.extend(image_path)
    acc = 0
    for i in range(len(test_score_list)):
        cls1 = 1 if test_score_list[i] > 5 else 0
        cls2 = 1 if test_output_list[i] > 5 else 0
        if cls1 == cls2:
            acc += 1
    test_plcc = calculate_plcc(test_score_list, test_output_list).item()
    test_srcc = calculate_srcc(test_score_list, test_output_list).item()
    print(f'Test Result ... PLCC: {test_plcc}, SRCC: {test_srcc}, Accuracy: {acc/len(test_output_list)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=34)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    train_dataset = BBDataset(file_dir='', type='train', test=False)
    valid_dataset = BBTestDataset(file_dir='', type='validation', test=True)
    test_dataset = BBTestDataset(file_dir='', type='test', test=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_model = BertModel.from_pretrained('bert-base-uncased')
    text_model = nn.DataParallel(text_model)
    text_model = text_model.to(device)
    lita_model = LITAModel()
    lita_model = nn.DataParallel(lita_model)
    lita_model = lita_model.to(device)
    optimizer = torch.optim.Adam(list(lita_model.module.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    train(args, text_model, lita_model)
    test(args, lita_model)


import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from PIL import Image
import json
import os

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Options:
    # Dataset parameters
    train_root = 'dataset/'  # Path to training images
    train_annotations = 'dataset/annotations.json'
    character = string.printable[:-6]  # Add all characters present in your dataset
    
    # Training parameters
    saved_model = ''  # Path to pretrained model (if continuing training)
    batch_size = 64
    workers = 4
    num_epochs = 100
    lr = 0.001
    save_interval = 10
    save_dir = './saved_models/'
    validation_split = 0.1  # 10% for validation
    
    # Model architecture (keep original TPS configuration)
    Transformation = 'TPS'
    FeatureExtraction = 'ResNet'
    SequenceModeling = 'BiLSTM'
    Prediction = 'Attn'
    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    batch_max_length = 25
    imgH = 32
    imgW = 100
    PAD = True
    rgb = False
    sensitive = False  # Set to True if using case-sensitive
    
    def __init__(self):
        if self.sensitive:
            self.character = string.printable[:-6]

opt = Options()

class CustomDataset(Dataset):
    def __init__(self, annotations_path, img_root, opt):
        with open(annotations_path) as f:
            self.annotations = [ann for ann in json.load(f) if ' ' not in ann['text']]  # Filter here
        self.img_root = img_root
        self.opt = opt
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        img_path = os.path.join(self.img_root, annotation['filename'])
        image = Image.open(img_path).convert('RGB' if self.opt.rgb else 'L')
        return image, annotation['text']

def train(opt):
    # Create save directory
    os.makedirs(opt.save_dir, exist_ok=True)

    # Prepare converter
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    # Create model
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Load pretrained weights
    if opt.saved_model:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # Loss and optimizer
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Prepare datasets
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    # Load full dataset and split
    full_dataset = CustomDataset(opt.train_annotations, opt.train_root, opt)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * opt.validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        collate_fn=align_collate,
        pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=align_collate,
        pin_memory=True)

    # Training loop
    best_accuracy = 0
    for epoch in range(opt.num_epochs):
        model.train()
        total_loss = 0
        
        for images, texts in train_loader:
            batch_size = images.size(0)
            print("batch, ", batch_size)
            images = images.to(device)
            
            # Prepare text for loss
            text_for_loss, length_for_loss = converter.encode(texts, batch_max_length=opt.batch_max_length)
            
            # Forward pass
            if 'CTC' in opt.Prediction:
                preds = model(images, text_for_loss)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                loss = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
            else:
                preds = model(images, text_for_loss[:, :-1], is_train=True)
                loss = criterion(preds.view(-1, preds.size(-1)), text_for_loss[:, 1:].contiguous().view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        avg_loss = total_loss / len(train_loader)
        current_accuracy = validate(model, val_loader, converter, opt)
        print(f'Epoch [{epoch+1}/{opt.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {current_accuracy:.2f}%')

        # Save model
        if (epoch+1) % opt.save_interval == 0:
            torch.save(model.state_dict(), f'{opt.save_dir}/epoch_{epoch+1}.pth')
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), f'{opt.save_dir}/best_accuracy.pth')

def validate(model, val_loader, converter, opt):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts in val_loader:
            batch_size = images.size(0)
            images = images.to(device)
            
            if 'CTC' in opt.Prediction:
                preds = model(images, None)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                preds = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                preds_str = [pred.split('[s]')[0] for pred in preds_str]
            
            for pred, true_text in zip(preds_str, texts):
                if pred == true_text:
                    correct += 1
                total += 1
    
    model.train()
    return (correct / total) * 100 if total > 0 else 0

if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = True
    train(opt)
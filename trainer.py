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
    character = string.printable[:-6] + ' '  # Added space character
    
    # Training parameters
    saved_model = './saved_models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'  # Path to pretrained model (if continuing training)
    batch_size = 64
    workers = 4
    num_epochs = 100
    lr = 0.001
    save_interval = 10
    save_dir = './saved_models/'
    validation_split = 0.1  # 10% for validation
    
    # Model architecture
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
    sensitive = True  # Case-sensitive handling

    def __init__(self):
        if self.sensitive:
            self.character = string.printable[:-6] + ' '

opt = Options()

class CustomDataset(Dataset):
    def __init__(self, annotations_path, img_root, opt):
        with open(annotations_path) as f:
            self.annotations = json.load(f)  # Removed space filter
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
    os.makedirs(opt.save_dir, exist_ok=True)

    # Prepare converter with updated character set
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    # Create model with new character count
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Handle pretrained weights with new space class
    if opt.saved_model:
        print(f'Loading pretrained weights from {opt.saved_model}')
        pretrained = torch.load(opt.saved_model, map_location=device)
        model_dict = model.state_dict()

        # 1. Handle embedding layer expansion
        embed_key = 'module.Prediction.embedding.weight'
        if embed_key in pretrained:
            old_embed_size, embed_dim = pretrained[embed_key].shape
            new_embed_size = model_dict[embed_key].shape[0]
            
            if old_embed_size != new_embed_size:
                print(f"Adjusting embedding layer from {old_embed_size} to {new_embed_size}")
                model_dict[embed_key][:old_embed_size] = pretrained[embed_key]
                pretrained[embed_key] = model_dict[embed_key]

        # 2. Handle attention RNN layers
        attention_layers = {
            'weight_ih': (0, 1),  # (output_dim, input_dim)
            'weight_hh': (0, 1),
            'bias_ih': (0,),
            'bias_hh': (0,)
        }
        
        base_path = 'module.Prediction.attention_cell.rnn.'
        for layer in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
            key = base_path + layer
            if key in pretrained:
                old_tensor = pretrained[key]
                new_tensor = model_dict[key]
                
                # Handle weight dimensions
                if len(old_tensor.shape) == 2:
                    min_dim0 = min(old_tensor.shape[0], new_tensor.shape[0])
                    min_dim1 = min(old_tensor.shape[1], new_tensor.shape[1])
                    new_tensor[:min_dim0, :min_dim1] = old_tensor[:min_dim0, :min_dim1]
                # Handle bias dimensions
                else:  
                    min_dim0 = min(old_tensor.shape[0], new_tensor.shape[0])
                    new_tensor[:min_dim0] = old_tensor[:min_dim0]
                
                pretrained[key] = new_tensor
                print(f"Adjusted {key} dimensions")

        # 3. Handle projection layer if exists
        proj_key = 'module.Prediction.attention_cell.linear_proj.weight'
        if proj_key in pretrained:
            old_proj = pretrained[proj_key]
            new_proj = model_dict[proj_key]
            min_dim = min(old_proj.shape[1], new_proj.shape[1])
            new_proj[:, :min_dim] = old_proj[:, :min_dim]
            pretrained[proj_key] = new_proj
            print("Adjusted projection layer")

        # 4. Handle final generator layer
        gen_weight_key = 'module.Prediction.generator.weight'
        gen_bias_key = 'module.Prediction.generator.bias'
        
        if gen_weight_key in pretrained:
            old_num_class = pretrained[gen_weight_key].size(0)
            new_num_class = opt.num_class
            
            model_dict[gen_weight_key][:old_num_class] = pretrained[gen_weight_key]
            model_dict[gen_bias_key][:old_num_class] = pretrained[gen_bias_key]
            pretrained[gen_weight_key] = model_dict[gen_weight_key]
            pretrained[gen_bias_key] = model_dict[gen_bias_key]

        # 5. Load modified weights
        model.load_state_dict(pretrained, strict=False)
        print('Successfully loaded pretrained weights with full dimension adjustments')

    # Loss and optimizer
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Prepare datasets
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    full_dataset = CustomDataset(opt.train_annotations, opt.train_root, opt)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * opt.validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
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
    print("starting training")
    best_accuracy = 0
    for epoch in range(opt.num_epochs):
        model.train()
        total_loss = 0
        
        for images, texts in train_loader:
            batch_size = images.size(0)
            print("batch", batch_size)
            images = images.to(device)
            
            # Prepare text with space characters
            text_for_loss, length_for_loss = converter.encode(texts, batch_max_length=opt.batch_max_length)
            
            if 'CTC' in opt.Prediction:
                preds = model(images, text_for_loss)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                loss = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
            else:
                preds = model(images, text_for_loss[:, :-1], is_train=True)
                loss = criterion(preds.view(-1, preds.size(-1)), text_for_loss[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.item()
            print("loss: ", loss.item())

        # Validation with space support
        avg_loss = total_loss / len(train_loader)
        current_accuracy = validate(model, val_loader, converter, opt)
        print(f'Epoch [{epoch+1}/{opt.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {current_accuracy:.2f}%')

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
            
            # Space-aware validation
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
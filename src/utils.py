import torch
import torch.nn as nn
from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs, targets = probs.view(-1), targets.view(-1)
        intersection = (probs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    scaler = torch.amp.GradScaler('cuda') # 初始化 Scaler
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        # 使用 autocast 開啟混合精度
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
        
        # 縮放梯度並更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # 計算 Dice
            preds, masks = preds.view(-1), masks.view(-1)
            intersection = (preds * masks).sum()
            dice = (2. * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
            dice_scores.append(dice.item())
    return sum(dice_scores) / len(dice_scores)
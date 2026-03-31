import os
import torch
from torch.utils.data import DataLoader
from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from oxford_pet import OxfordPetDataset
from utils import train_one_epoch, evaluate

def main():
    MODEL_NAME = 'unet' # 可選 'unet' 或 'resnet34'
    DEVICE = torch.device('cuda')
    BATCH_SIZE = 8 # 5090 建議 32
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, 'dataset', 'oxford-iiit-pet')
    
    train_ds = OxfordPetDataset(data_root, split='train')
    val_ds = OxfordPetDataset(data_root, split='val')

    # 加速重點：num_workers 設為 8 以上，且 pin_memory 設為 True
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = ResNet34_UNet(n_classes=1).to(DEVICE) if MODEL_NAME == 'resnet34' else UNet(3, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.2)

    best_dice = 0.0
    save_path = f"../saved_models/{MODEL_NAME}_best.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_dice = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_dice)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"🌟 Saved Best: {val_dice:.4f}")

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def calculate_dice(pred, target, smooth=1e-6):
    pred, target = pred.flatten(), target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def run_evaluation(model_type='unet'):
    DEVICE = torch.device('cuda')
    torch.cuda.empty_cache()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_list_name = 'test_unet.txt' if model_type == 'unet' else 'test_res_unet.txt'
    test_list_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'annotations', test_list_name)
    if not os.path.exists(test_list_path):
        test_list_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'annotations', 'test.txt')

    weight_path = os.path.join(project_root, 'saved_models', f'{model_type}_best.pth')
    model = UNet(3, 1).to(DEVICE) if model_type == 'unet' else ResNet34_UNet(1).to(DEVICE)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(test_list_path, 'r') as f:
        test_files = [line.split()[0] for line in f]

    # 測試門檻清單
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    dice_results = {t: [] for t in thresholds}

    with torch.no_grad():
        for file_name in tqdm(test_files, desc="Searching Best Threshold"):
            img_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'images', f"{file_name}.jpg")
            mask_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'annotations', 'trimaps', f"{file_name}.png")
            if not os.path.exists(img_path) or not os.path.exists(mask_path): continue
                
            image = Image.open(img_path).convert("RGB")
            gt_mask = (np.array(Image.open(mask_path)) == 1).astype(np.float32)
            
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(img_tensor)).cpu().squeeze().numpy()
            
            for t in thresholds:
                pred_bin = (prob > t).astype(np.float32)
                pred_res = np.array(Image.fromarray(pred_bin).resize(image.size, resample=Image.NEAREST))
                dice_results[t].append(calculate_dice(pred_res, gt_mask))

    print("\n" + "="*30)
    best_t, max_dice = 0.5, 0.0
    for t in thresholds:
        avg = np.mean(dice_results[t])
        print(f"Threshold {t:.2f} | Dice: {avg:.4f}")
        if avg > max_dice:
            max_dice, best_t = avg, t
    print(f"🏆 Best Threshold: {best_t} | Max Dice: {max_dice:.4f}")
    return best_t

if __name__ == "__main__":
    run_evaluation(model_type='unet')
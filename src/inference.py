import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def rle_encode(mask):
    pixels = mask.T.flatten() 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_inference(model_type='unet', BEST_T=0.45):
    DEVICE = torch.device('cuda')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    test_list_name = 'test_unet.txt' if model_type == 'unet' else 'test_res_unet.txt'
    test_list_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'annotations', test_list_name)
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

    results = []
    with torch.no_grad():
        for file_name in tqdm(test_files, desc="TTA Inference"):
            img_path = os.path.join(project_root, 'dataset', 'oxford-iiit-pet', 'images', f"{file_name}.jpg")
            image = Image.open(img_path).convert("RGB")
            ori_w, ori_h = image.size

            # TTA: 原圖 + 水平翻轉
            img_t = transform(image).unsqueeze(0).to(DEVICE)
            prob_n = torch.sigmoid(model(img_t))
            
            img_f = transform(transforms.functional.hflip(image)).unsqueeze(0).to(DEVICE)
            prob_f = torch.sigmoid(model(img_f))
            prob_f_back = transforms.functional.hflip(prob_f)
            
            # 平均機率圖
            final_prob = ((prob_n + prob_f_back) / 2).cpu().squeeze().numpy()
            binary_mask = (final_prob > BEST_T).astype(np.uint8)

            final_mask = np.array(Image.fromarray(binary_mask).resize((ori_w, ori_h), resample=Image.NEAREST))
            results.append({"image_id": file_name, "encoded_mask": rle_encode(final_mask)})

    pd.DataFrame(results).to_csv(f"submission_{model_type}_final.csv", index=False)
    print(f"✅ CSV Generated with TTA & Threshold {BEST_T}")

if __name__ == "__main__":
    # 假設 evaluate 跑出來 0.45 最高，就填 0.45
    run_inference(model_type='unet', BEST_T=0.45)
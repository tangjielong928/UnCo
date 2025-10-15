import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def extract_clip_features(image_path, bounding_boxes, model, processor, device):
    image = Image.open(image_path).convert('RGB')
    region_features = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)
        region = image.crop((x1, y1, x2, y2))
        inputs = processor(images=region, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        with torch.no_grad():
            feat = model.get_image_features(pixel_values)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            region_features.append(feat.cpu().numpy())
    region_features = np.concatenate(region_features, axis=0)
    return region_features

def process_folder(image_folder, bbox_folder, output_folder, device='cuda'):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_id = os.path.splitext(fname)[0]
        image_path = os.path.join(image_folder, fname)
        bbox_path = os.path.join(bbox_folder, img_id + '.npy')
        if not os.path.exists(bbox_path):
            continue
        bounding_boxes = np.load(bbox_path)  # shape: [N,4]
        features = extract_clip_features(image_path, bounding_boxes, model, processor, device)
        out_path = os.path.join(output_folder, img_id + '_clipfeat.npy')
        np.save(out_path, features)
        print(f'Saved {out_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--bbox_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    process_folder(args.image_folder, args.bbox_folder, args.output_folder, device=args.device)

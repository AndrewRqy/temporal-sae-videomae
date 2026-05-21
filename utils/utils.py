from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
import torch.nn as nn
from models.clip import Clip
from models.dino import Dino
from models.siglip import Siglip
from models.videomae import VideoMAE
from datasets.ssv2 import SSv2Dataset
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection

class ImageCollate:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        images = [img[0] for img in batch]
        return self.processor(images=images, return_tensors="pt", padding=True)

class VideoMAECollate:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        frames_list = [item[0] for item in batch]
        return self.processor(images=frames_list, return_tensors="pt")

class DinoSSv2Collate:
    """Flatten SSv2 video frames as individual images for DINOv2."""
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        all_frames = []
        for frames_list, _ in batch:
            all_frames.extend(frames_list)
        return self.processor(images=all_frames, return_tensors="pt")

def get_collate_fn(processor):
    return ImageCollate(processor)

def get_videomae_collate_fn(processor):
    return VideoMAECollate(processor)

def get_dataset(args, preprocess, processor, split, subset=1.0):
    if args.dataset_name == 'cc3m':
        raise NotImplementedError
    elif args.dataset_name == 'inat_birds':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'inat':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'imagenet':
        ds = ImageNet(root=args.data_path, split=split, transform=preprocess)
    elif args.dataset_name == 'cub':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'ssv2':
        ds = SSv2Dataset(args.data_path, split=split)
        keep_every = int(1.0 / subset)
        if keep_every > 1:
            ds = Subset(ds, list(range(0, len(ds), keep_every)))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=get_videomae_collate_fn(processor))
        return ds, dl
    elif args.dataset_name == 'ssv2_dino':
        ds = SSv2Dataset(args.data_path, split=split)
        keep_every = int(1.0 / subset)
        if keep_every > 1:
            ds = Subset(ds, list(range(0, len(ds), keep_every)))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=DinoSSv2Collate(processor))
        return ds, dl

    keep_every = int(1.0 / subset)
    if keep_every > 1:
        ds = Subset(ds, list(range(0, len(ds), keep_every)))
    if processor is not None:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=get_collate_fn(processor))
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return ds, dl

def get_model(args):
    if args.model_name.startswith('clip'):
        clip = Clip(args.model_name, args.device)
        return clip, clip.processor
    elif args.model_name.startswith('dino'):
        dino = Dino(args.model_name, args.device)
        return dino, dino.processor
    elif args.model_name.startswith('siglip'):
        siglip = Siglip(args.model_name, args.device)
        return siglip, siglip.processor
    elif 'videomae' in args.model_name:
        vm = VideoMAE(args.model_name, args.device)
        return vm, vm.processor

def get_text_model(args):
    if args.model_name.startswith('clip'):
        model = CLIPTextModelWithProjection.from_pretrained(f"openai/{args.model_name}").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(f"openai/{args.model_name}")
        return model, tokenizer

class IdentitySAE(nn.Module):
    def encode(self, x):
        return x
    def decode(self, x):
        return x

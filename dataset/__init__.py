import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, coco_dataset

from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
     
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    

    if dataset=='re':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset
    elif dataset== 'coco':
        train_dataset = coco_dataset(config['train_file'], train_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=True, add_object=config['add_object'])
        val_dataset = coco_dataset(config['val_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        test_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset  
    else:
        raise NotImplementedError
    

def coco_collate_fn(batch):
    image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, object_label, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        object_labels.append(object_label)
    return torch.stack(image_list,dim=0), caption_list, object_labels, image_id_list, gold_caption_list


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

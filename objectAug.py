import json
import cv2
import matplotlib.pyplot as plt

def objectAug:
    dataset_path  = '../input/data'
    annot = dataset_path +'/train_all.json'
    
    with open(annot,'r') as f:
        dataset = json.loads(f.read())
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    
    

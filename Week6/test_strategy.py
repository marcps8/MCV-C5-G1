import os, sys
import collections
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchinfo # to print model summary
from torchinfo import summary # to print model summary
from tqdm.auto import tqdm # used in train function
import torchvision # print model image
from torchview import draw_graph # print model image
import random
from PIL import Image
import glob
from pathlib import Path
from timeit import default_timer as timer  
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

from model.augmentation_InceptionResnetV1 import walk_through_dir, transform_data, get_data_sets_path, loadImageData, detail_one_sample_data, myDataLoader

IMG_EMBEDDINGS = "/ghome/group01/MCV-C5-G1/Week6/feature_embeddings/image/"
MODEL = "/ghome/group01/MCV-C5-G1/Week6/model/best-model-parameters-undersampled.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"   


def combine_id_test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, test_data, transormer):
    
    predictions = [['VideoName','ground_truth','prediction']]
    video_predictions = collections.defaultdict(list)
    
    
    for f in test_data.imgs:
        f_name = f[0]
        f_name = f_name[len(f_name)-19:len(f_name)-4]+'.mp4'
        user_id = f_name.split(".")[0]
        cat = f[1]+1 # f[1] from 0 to 6
        predictions.append([f_name,cat,'-1'])
        #print(predictions[-1])

    # Put model in test mode
    model.eval()
    
    # Setup test loss and test accuracy values
    test_acc = 0
    
    print("Evaluating on test set...")
    # Loop through data loader data batches
    counter = 1
    for batch, (X, y) in enumerate(dataloader):

        print(f"Batch: {batch} of {len(dataloader)}")
        # Send data to target device
        X, y = X.to(device), y.to(device)

        if(len(y)!=1):
            print("ERROR: batch size (test stage) =", len(y))
            print("ERROR: my_test_step function require batch size = 1 to generate the correct output file")
            exit()
        
        # 1. Forward pass
        y_pred = model(X)

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        test_acc += (y_pred_class == y).sum().item()/len(y_pred)

        predictions[counter][2] = y_pred_class.tolist()[0]+1 # y_pred_class.tolist()[0] from 0 to 6
        user_id = predictions[counter][0].split(".")[0]
        video_predictions[user_id].append(predictions[counter])
        counter +=1
    
    # Adjust metrics to get average accuracy per batch 
    test_acc = test_acc / len(dataloader)
    print("Average accuracy = ", test_acc)
    
    original_predictions = predictions.copy()
    correct = 0
    t = 0
    for sample in original_predictions[1:]:
        if sample[1] == sample[2]:
            correct += 1
        t += 1

    accuracy = correct / t
    print(f"Average Accuracy: {accuracy}")
    
    np.savetxt("evaluation/predictions_test_set.csv",
        original_predictions,
        delimiter =",",
        fmt ='% s')
    
    for _, preds in video_predictions.items():
        prediction = [pred[2] for pred in preds]
        most_common_number = max(set(prediction), key=prediction.count)
        
        for pred in preds:
            pred[2] = most_common_number
    

    correct_predictions = 0
    total_predictions = 0
    # print(video_predictions)
    for _ , preds in video_predictions.items():
        for pred in preds:
            if pred[2] == pred[1]:
                correct_predictions += 1
            total_predictions += 1
            
    our_predictions = [['VideoName', 'ground_truth', 'prediction']]
    for user_id in video_predictions.keys():
        preds = video_predictions[user_id]
        our_predictions.extend(preds)
        
    accuracy = correct_predictions / total_predictions
    print(f"Combined Average Accuracy: {accuracy}")
    
    np.savetxt("evaluation/predictions_test_set_our.csv",
        our_predictions,
        delimiter =",",
        fmt ='% s')


def main(data_path,parameters_dict,class_weights):    

    # preliminaries
    walk_through_dir(data_path)
    train_dir, valid_dir, test_dir = get_data_sets_path(data_path)

    # data transformation
    data_train_transform, data_valid_test_transform = transform_data(parameters_dict['image_size']['values'][0], parameters_dict['image_size']['values'][1])
    
    # data loader
    train_data, valid_data, test_data, class_names = loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform)
    
    num_classes = len(class_names)
    detail_one_sample_data(train_data, class_names)
    train_dataloader, valid_dataloader, test_dataloader = myDataLoader(train_data, valid_data, test_data, parameters_dict['num_workers']['values'][0], parameters_dict['batch_size']['values'][0], parameters_dict['batch_size_valid']['values'][0], parameters_dict['batch_size_test']['values'][0])

    # model definition
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=num_classes).to(device)
    
    checkpoint = torch.load(MODEL, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    combine_id_test_step(model, test_dataloader, test_data, data_valid_test_transform)

    
        
if __name__ == "__main__":

    parameters_dict = {
        'image_size': {
            'values': [224, 224]
            },
        'num_workers': {
            'values': [0]
            },
        'batch_size': {
            'values': [256]
            },
        'batch_size_valid': {
            'values': [256]
            },
        'batch_size_test': {
            'values': [1]
            },
        'num_epochs': {
            'values': [300]
            },
        'learning_rate': {
            'values': [1e-6]
            },
        'early_stopping': {
            'values': [50]
            },
    }

    # train data distribution per category [10, 164, 1264, 2932, 1353, 232, 51]
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device) # can be changed to use class weights
    
    # data_path = sys.argv[1] # path to the input data
    data_path = './data'

    main(data_path, parameters_dict,class_weights)
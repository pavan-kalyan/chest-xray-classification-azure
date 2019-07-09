
import torch
import torch.nn as nn
from torchvision import transforms
import json
import pretrainedmodels
import os, uuid, sys
from PIL import Image
from io import BytesIO
from azure.storage.blob import BlockBlobService, PublicAccess
import pretrainedmodels.utils as utils
from collections import OrderedDict
import base64
import logging
import time


DISEASE_LIST = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

from azureml.core.model import Model
global img_string
img_string = ''

# helper functions
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    buff.seek(0)
    return Image.open(buff)

def init():
    global model
    seed = 0
    torch.manual_seed(seed)
    logging.basicConfig(level=logging.DEBUG)
    model_name = 'nasnetalarge'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='')
    model_path = Model.get_model_path('full_chexray_e17')
    
    # if you want to test the run script on your local system you can uncomment the following line and comment the one above
    #model_path = os.path.join(base_path,'output/chexray_nasnet_e10.pth.tar')
    n_classes=14
    
    in_ftrs=model.last_linear.in_features
    model.last_linear=nn.Sequential(
    nn.Linear(in_ftrs,256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(256,n_classes),
    nn.Sigmoid())
    
    chkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = chkpt['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)
    global load_img
    load_img = utils.LoadImage()
    global credentials
    credentials = {
        "datastore_name" : "chexrayds",
        "container_name" : "chexray-images",
        "account_name" : "chexray14",
        "account_key" : "rXogWjkm8lXV0QfHAety1KktKGxbb1pMGoWkd4E9FI3QN+MW2qp4bWlFRymGS2zR0MyroV+VJcgHurr8+dhQdg=="
    } 
    model.eval()


def run(input_data):
    #print(input_data)
    imgstring = json.loads(input_data)['data']
    img_name = json.loads(input_data)['file_name']
    should_upload = json.loads(input_data)['upload']
    try:
        print(img_name)
        IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_RGB_SD = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)

        # the following transformation is applied on each incoming message.
        transform=transforms.Compose([
                                 transforms.Resize((331,331),interpolation=Image.NEAREST),
                                 transforms.ToTensor(),  
                                 normalize])

        img = b64_2_img(imgstring)
        
        # upload the image to blob storage, if sufficient permission is given.
        if should_upload == True:
            base_path = os.getcwd()
            

            timestr=time.strftime("%Y%m%d-%H%M%S")
            file_name = img_name + timestr

            full_path_to_file = os.path.join(base_path, file_name)
            img.save(file_name +'.png','PNG')
            block_blob_service = BlockBlobService(account_name = credentials['account_name'], account_key = credentials['account_key']) 
            block_blob_service.create_blob_from_path(credentials['container_name'], 'images/'+file_name+'.png', full_path_to_file+'.png')
            print('uploaded '+ file_name+'.png')
            os.remove(file_name+'.png')
            
        # get prediction
        with torch.no_grad():
            input_tensor = transform(img.convert('RGB'))
            input_tensor = input_tensor.unsqueeze(0)
            print(input_tensor.shape)
            output = model(input_tensor)
            vals,ids = output.topk(5)
            data = {}
            labels = []
            for id in ids.tolist()[0]:
                labels.append(DISEASE_LIST[id])
            print(labels)
            data['labels'] = labels;
            print(vals.tolist()[0])
            data['probabilities'] = vals.tolist()[0]
            total_sum = 0
            for val in vals.tolist()[0]:
                total_sum = total_sum + val
            print(total_sum)



        result = {
            "labels": labels,
            "probabilities": vals.tolist()[0],
            "comments": "",
            "prob_sum": total_sum
        }
    
    except Exception as e:
        print("Unexpected error:", str(e))
        result = {
            "comments": str(e), 
        }
        return result
    
    
    print(result)
    return result

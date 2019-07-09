#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('ls')
get_ipython().system('pwd')


# In[2]:



# Allow multiple displays per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import azureml.core
import os,sys
import numpy as np
import azureml.data
from azureml.data.data_reference import DataReference

print(azureml.core.VERSION)


# In[4]:



from azureml.core import Workspace, Datastore

ws = Workspace.from_config()


# In[5]:


# Print workspace config details.
print(ws.name)


# In[6]:


base_path = os.path.dirname(os.getcwd())
data_raw_dir = os.path.join(base_path,'data/raw')
data_proc_dir = os.path.join(base_path,'data/processed')
data_img_dir = os.path.join(base_path,'data/processed/images')
label_file = os.path.join(data_raw_dir,'Data_Entry_2017.csv')
notshared_dir = os.path.join(base_path,'notshared')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


import pickle
import random
import re
import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn.model_selection


# In[8]:


labels_df = pd.read_csv(label_file)



#total_patient_number = labels_df['Patient ID'].nunique()
total_patient_number = 60
NIH_patients_and_labels_file = 'Data_Entry_2017.csv'
NIH_annotated_file = 'BBox_List_2017.csv' # exclude from train pathology annotated by radiologists 
bad_imgs_file = 'blacklist.txt'# exclude what viusally looks like bad images

patient_id_original = [i for i in range(1,total_patient_number + 1)]

# ignored images list is used later, since this is not a patient ID level issue

ignored_imgs_ids = []
bad_imgs = open(os.path.join(data_raw_dir,'blacklist.txt'))
ignored_imgs_ids = bad_imgs.read().splitlines()
bad_imgs.close()

bbox_df = pd.read_csv(os.path.join(data_raw_dir,NIH_annotated_file))
bbox_patient_index_df = bbox_df['Image Index'].str.slice(3, 8)

bbox_patient_index_list = []
for index, item in bbox_patient_index_df.iteritems():
    bbox_patient_index_list.append(int(item))

patient_id = list(set(patient_id_original) - set(bbox_patient_index_list))
print("len of original patient id is", len(patient_id_original))
print("len of cleaned patient id is", len(patient_id))
print("len of unique patient id with annotated data", 
      len(list(set(bbox_patient_index_list))))
print("len of patient id with annotated data",bbox_df.shape[0])


# In[9]:


random.seed(0)
random.shuffle(patient_id)

print("first ten patient ids are", patient_id[:10])

# training:valid:test=7:1:2
patient_id_train = patient_id[:int(total_patient_number * 0.7)]
patient_id_valid = patient_id[int(total_patient_number * 0.7):int(total_patient_number * 0.8)]
# get the rest of the patient_id as the test set
patient_id_test = patient_id[int(total_patient_number * 0.8):]
patient_id_test.extend(bbox_patient_index_list)
patient_id_test = list(set(patient_id_test))[:50]


print("train:{} valid:{} test:{}".format(len(patient_id_train), len(patient_id_valid), len(patient_id_test)))


# In[ ]:





# In[ ]:





# In[10]:


labels_df.tail()
labels_df['Finding Labels'].nunique()


# In[ ]:





# In[11]:


index = np.arange(len(labels_df['Image Index']))
print(index)


# In[12]:


dummy = labels_df['Finding Labels'].str.split( '|', expand=False).str.join(sep='*').str.get_dummies(sep='*')
x_axis = list(dummy.columns)
y_axis = list(dummy.sum())
plt.title('distribution of diseases in dataset')
plt.xticks(list(range(1,16)),x_axis,rotation='vertical')
plt.bar(list(range(1,16)),y_axis)


# In[13]:


DISEASE_LIST = x_axis
DISEASE_LIST.remove('No Finding')
for id,ele in enumerate(DISEASE_LIST):
    print(str(id) + ' - '+ ele)


# In[14]:


def generate_labels_and_indices(df, patient_ids,ignored_ids):
    img_name_ids = []
    img_labels = {}
    for patient in tqdm.tqdm(patient_ids):
        for _, row in df[df['Patient ID'] == patient].iterrows():
            img_idx = row['Image Index']
            if img_idx not in ignored_ids:
                img_name_ids.append(img_idx)
                img_labels[img_idx] = np.zeros(14, dtype=np.uint8)
                for disease_idx,disease_name in enumerate(DISEASE_LIST):
                    if disease_name in row['Finding Labels'].split('|'):
                        img_labels[img_idx][disease_idx] = 1
                    else:
                        img_labels[img_idx][disease_idx] = 0
    return img_name_ids, img_labels


# In[15]:


train_data_index, train_labels = generate_labels_and_indices(labels_df, patient_id_train,ignored_imgs_ids)
valid_data_index, valid_labels = generate_labels_and_indices(labels_df, patient_id_valid,ignored_imgs_ids)
test_data_index, test_labels = generate_labels_and_indices(labels_df, patient_id_test,ignored_imgs_ids)


# In[16]:


type(train_labels)


# In[17]:


{k: train_labels[k] for k in list(train_labels)[:5]}


# In[18]:


#checking one sample
labels_df[labels_df['Image Index'] == '00000024_000.png']


# In[ ]:





# In[19]:


print("train, valid, test image number is:", len(train_data_index), len(valid_data_index), len(test_data_index))

# save the data
labels_all = {}
labels_all.update(train_labels)
labels_all.update(valid_labels)
labels_all.update(test_labels)

partition_dict = {'train': train_data_index, 'test': test_data_index, 'valid': valid_data_index}

with open(os.path.join(data_proc_dir,'labels14_unormalized_cleaned.pickle'), 'wb') as f:
    pickle.dump(labels_all, f)

with open(os.path.join(data_proc_dir,'partition14_unormalized_cleaned.pickle'), 'wb') as f:
    pickle.dump(partition_dict, f)
    
# also save the patient id partitions for pytorch training    
with open(os.path.join(data_proc_dir,'train_test_valid_data_partitions.pickle'), 'wb') as f:
    pickle.dump([patient_id_train,patient_id_valid,
                 patient_id_test,
                list(set(bbox_patient_index_list))], f)


# In[20]:


get_ipython().system('ls data/processed/')
get_ipython().system('ls')


# In[ ]:





# In[ ]:





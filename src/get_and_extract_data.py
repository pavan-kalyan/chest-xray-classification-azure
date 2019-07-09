#!/usr/bin/env python
# coding: utf-8

# In[35]:


import azureml.core
import os,sys
import numpy as np
import json
import azureml.data
from azureml.data.data_reference import DataReference
print(azureml.core.VERSION)

from azureml.core import Workspace, Datastore

ws = Workspace.from_config()



from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[36]:


# git statements to get entire code structure


# In[ ]:





# In[37]:


base_path = os.path.dirname(os.getcwd())
data_raw_dir = os.path.join(base_path,'data/raw')
data_proc_dir = os.path.join(base_path,'data/processed')
data_img_dir = os.path.join(base_path,'data/processed/images')
label_file = os.path.join(data_raw_dir,'Data_Entry_2017.csv')
notshared_dir = os.path.join(base_path,'notshared')


# In[38]:


datastores = ws.datastores
for name, ds in datastores.items():
    print(name, ds.datastore_type)
#ws.set_default_datastore('chexrayds')


# In[48]:


with open(os.path.join(notshared_dir,'credentials.json')) as creds:    
    credentials = json.load(creds)
    
#print(credentials)
ds = Datastore.register_azure_blob_container(workspace=ws, 
                                             datastore_name=credentials['datastore_name'], 
                                             container_name=credentials['container_name'],
                                             account_name=credentials['account_name'], 
                                             account_key=credentials['account_key'],
                                             create_if_not_exists=False)


# In[49]:


ds = Datastore.get(ws,datastore_name='chexrayds')
print(ds.name)


# In[ ]:





# In[50]:


import azureml.data
from azureml.data.data_reference import DataReference
ws.set_default_datastore('chexrayds')
ds.as_mount()


# In[54]:


ds.download(data_raw_dir,None,True)


# In[ ]:





# In[ ]:





# In[43]:


import tarfile


# In[ ]:





# In[45]:


# proper code to extract data into images

for filename in os.listdir(data_raw_dir):
    if filename.endswith(".tar"):
        tf=tarfile.open(os.path.join(*[data_raw_dir,filename,filename]),'w')
        # tf.extractall(base_path+'/data/raw/')
        print(filename)
        


# In[ ]:


#


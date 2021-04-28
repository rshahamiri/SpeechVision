#!/usr/bin/env python
# coding: utf-8

# In[1]:


import DysarthricSpeechVision as sv
# IMPORTANT: Deleting .ipynb_checkpoints from image paths
get_ipython().system("find '.' -name '*.ipynb_checkpoints' -exec rm -r {} +")


# In[4]:


# Enable this if you want to train the model for this speaker from scracth. 
# Otherwise, the previously trained model is continued training.
# This loads the base, control model.
sv.training_restart_initalize()


# In[5]:


# Train
sv.set_gpus("1")
sv.train(is_dnn_structure_changned= False, enabled_trasfer_learning=False, max_epoch=5000)


# In[ ]:


sv.visualize_training()


# In[ ]:


# Test
sv.set_gpus("1")
sv.test_generator(sv.test_set)


# In[ ]:





# Selected cnn-control-testing82percent with 0.5 droprate for all layers.
# Control Training Accuracy: 92%
# Control Validation Accuracy: 84.87
# Control Testing Accuracy: 81.64
# You must set speaker_name
import utilities
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout , SpatialDropout2D
from keras.layers import AveragePooling2D, Activation
from keras.callbacks import History
import numpy as np
import pandas as pd
import os
import keras
from  keras import optimizers
from keras import losses
from keras.regularizers import l1_l2, l1,l2
from keras.models import model_from_json
from sklearn.utils import class_weight
import pyprog
import os

def set_gpus(gpus_number="1,2"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_number
    
SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))

speaker_name=input("Which speaker do you want to train/test? ")
train_set_path = SETTINGS_DIR+'/images/Dysarthric/Train/'+speaker_name
test_set_path = SETTINGS_DIR+"/images/Dysarthric/Test/"+speaker_name
dnn_file_name_structure = SETTINGS_DIR +"/Models/cnn_"+speaker_name+".json"
training_dynamics_path = SETTINGS_DIR+'/Training Performance/TrainingDynamics'+speaker_name+'.csv'
dnn_file_name_weights = SETTINGS_DIR +  "/Models/cnn_weight_"+speaker_name+".h5"

batch_size=256
image_input_size=(150,150)
vocab_size = utilities.get_no_folders_in_path(test_set_path)
print ("Vocabulary Size:",vocab_size)

def model_compile(model):
    model.compile(loss=losses.categorical_crossentropy,
                          optimizer=optimizers.Adadelta(),
                          metrics=['accuracy'])
    
def get_model():

    droprate=0.5

    classifier = Sequential()

    classifier.add( Convolution2D(  filters=32, kernel_size=(3,3), 
                                  input_shape= (*image_input_size,3), 
                                  activation='relu')  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=32, kernel_size=(3,3),activation='relu'  )  )
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))


    classifier.add( Convolution2D(  filters=64, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=64, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))

    classifier.add( Convolution2D(  filters=128, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=128, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))

    classifier.add( Convolution2D(  filters=256, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=256, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(0.5))
    classifier.add(Dropout(droprate))

    classifier.add (Flatten( ) )

    classifier.add(Dense (units=vocab_size, activation='softmax' ))
    classifier.summary()

    return classifier

def predict_an_image(image_path, model):
    
    from tensorflow.keras.preprocessing import image

    test_image = image.load_img(image_path, target_size = image_input_size)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) 

    y_pred = model.predict_classes(test_image,batch_size)[0]
    classes =training_set.class_indices
    for key, value in classes.items():
        if value==y_pred:
            break       

    pred_key=utilities.dictionary .index [ utilities.dictionary  ['FILE NAME'] == key ] 
    predicted_word=utilities.dictionary .iloc[pred_key[0],0]
    # Get true label
    true_key=true_key=utilities.file_to_index(image_path)
    true_word = utilities.dictionary .iloc[true_key,0]
    #print("Predicted:",predicted_word,", True:",true_word)
    return predicted_word, true_word

def read_epoch():
    if os.path.exists(training_dynamics_path):
        
        # First check the csv file has headres and add then if missing
        try:
            training_dynamics=pd.read_csv(training_dynamics_path)
            training_dynamics["Epoch"][len(training_dynamics)-1]
        except:
            df = pd.read_csv(training_dynamics_path, header=None, index_col=None)
            df.columns = columns=["","Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
            df.to_csv(training_dynamics_path, index=False)
        training_dynamics=pd.read_csv(training_dynamics_path)               
        return training_dynamics["Epoch"][len(training_dynamics)-1]
        
    else:
        return 0

def load_model():
    # Loading the CNN
    json_file = open(dnn_file_name_structure, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(dnn_file_name_weights)
    model_compile(model)
    return model

def save_model(model,is_max_val_inclluded=False,max_val=None, ep=None):
    # Save/overwrite the model
    if (is_max_val_inclluded):
        json_file_name = SETTINGS_DIR+"/Models/cnn_"+speaker_name+"_"+str(max_val)+"_"+str(ep)+".json"
        wights_file_name = SETTINGS_DIR+"/Models/cnn_weight_"+speaker_name+"_"+str(max_val)+"_"+str(ep)+".h5"
        # Delete previously stored models for this speaker
        for directory, s, files in os.walk(SETTINGS_DIR+"/Models/"):
            for f in files:
                if speaker_name in f:
                    file_path=directory+"/"+f
                    os.remove(file_path)
    else:
        json_file_name = dnn_file_name_structure
        wights_file_name = dnn_file_name_weights
    
    model_json = model.to_json()
    with open(json_file_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(wights_file_name)
    
def save_training_dynamics(epoch,history,with_header=False):
    training_dynamics=pd.DataFrame(
        data = [ [epoch, history.history['loss'][0] ,  history.history['acc'][0],  
                history.history['val_loss'][0],  history.history['val_acc'][0] ]],
        columns=["Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
    )
    if (with_header):
        with open(training_dynamics_path, 'a') as csv_file:
            training_dynamics.to_csv(csv_file, header=True)
    else:
        with open(training_dynamics_path, 'a') as csv_file:
            training_dynamics.to_csv(csv_file, header=False)
            
def visualize_training():
    import matplotlib.pyplot as plt
    if (os.path.isfile(training_dynamics_path) == False ):
        print ("Training dynamics file is not found.")
        return
    try:
        training_dynamics=pd.read_csv(training_dynamics_path)
        loss_values = training_dynamics["TrainingLoss"]
        val_loss_values = training_dynamics["ValidationLoss"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # Ploting Accuracy
        loss_values = training_dynamics["TrainingAccuracy"]
        val_loss_values = training_dynamics["ValidationAccuracy"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
     
    except:
        df = pd.read_csv(training_dynamics_path, header=None, index_col=None)
        df.columns = ["","Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
        df.to_csv(training_dynamics_path, index=False)
        visualize_training()
    
def get_train_test_sets():
        from keras.preprocessing.image import ImageDataGenerator
        
        # https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
        train_datagen = ImageDataGenerator(
                    rescale=1./255,
            width_shift_range=0.30,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            horizontal_flip=False)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # If shuffle=False then the validation results will be different from classifier.predict_generator()
        print ("Setting training date...")
        training_set = train_datagen.flow_from_directory(
            train_set_path,
            target_size=image_input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)
        
        print ("Setting testing date...")
        test_set = test_datagen.flow_from_directory(
           test_set_path,
            target_size=image_input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        return training_set, test_set
    
def test_generator(test_set_generator):
    steps=test_set_generator.samples/batch_size
    model = load_model()

    y_pred = model.evaluate_generator(test_set_generator, steps = steps, verbose = 1)
    y_test = test_set_generator.classes
    correct_classifications=0
    for pred,label in zip(y_pred, y_test):
        if pred.argmax()==label:
            correct_classifications+=1
    print ("Loss:", y_pred[0])
    print ("Acuracy:", y_pred[1] *100,"%")
    return 

def manual_testing():
    model = load_model() 
    #test_path = SETTINGS_DIR+"/images/Control/Test"
    #test_path = SETTINGS_DIR+"/images/Dysarthric/Test/F05"
    #test_path = SETTINGS_DIR+"/images/Dysarthric/Test/M06"
    #test_path = SETTINGS_DIR+"/images/Dysarthric/Test/M10"
    #test_path = SETTINGS_DIR+"/images/Control/Test"
    test_path=test_set_path
    
    correct_classifications=0
    i=0
    prog = pyprog.ProgressBar("Predicting ", " Done", utilities.get_no_files_in_path(test_path))
    # Show the initial status
    prog.update()
    no_processed=i
    for directory, s, files in os.walk(test_path):
            for f in files:
                file_path=directory+"/"+f
                if ("jpg" in f):                
                    predicted_word, true_word = predict_an_image(file_path,model)
                    #print (predicted_word,true_word)
                    if (predicted_word==true_word):
                        correct_classifications+=1
                    i+=1
                    prog.set_stat(i)
                    prog.update()

    prog.end()
    print ("Testing acuracy:", correct_classifications/i *100,"%")
    
def train(ideal_loss=0.01, is_dnn_structure_changned=False, max_epoch=50, enabled_trasfer_learning=False):
        
        # Check if speaker_name is set
        if (speaker_name==""):
            print ("Please set speaker_name and try again.")
            return
            
        is_new_dnn=False
        
        history = History()
        
        print("=================================================")
        
        if (os.path.isfile(dnn_file_name_structure) and
                (os.path.isfile(dnn_file_name_weights)) and 
                (is_dnn_structure_changned == False)):
            # load the previosly trained DNN
            if (enabled_trasfer_learning):
                # Enable Transfer Learning
                print ("Transfer learning is enabled.")
                model = FreezeLayers(load_model(),top_unfrozen_layer_name="conv2d_7" ) 
            else:
                print ("Transer learning is disabled.")
                model = load_model()
            print("CNN is loaded.")
        else:
            # Create a new model
            model =  get_model()                    
            print("CNN is created")
            # Erase the training_dynamic_csv file
            if os.path.exists(training_dynamics_path):
                os.remove(training_dynamics_path)
            is_new_dnn=True
            model_compile(model)
        
        ep= read_epoch()+1
        PringFrozenLayers(model)
        model.fit_generator(
            training_set,
            steps_per_epoch=training_set.samples/batch_size, epochs=1,                            
                             validation_data=test_set,
                             validation_steps=test_set.samples/batch_size,
                             workers=10, 
                             max_queue_size=10,  callbacks=[history])
        
        save_training_dynamics(ep,history,with_header=is_new_dnn)
       
        max_val = history.history['val_acc'][0]
        
        while (history.history['loss'][0] >= ideal_loss):
            print("Epoch", ep)
            model.fit_generator(
            training_set,
            steps_per_epoch=training_set.samples/batch_size,epochs=1,
                             validation_data=test_set,
                             validation_steps=test_set.samples/batch_size,
                             workers=10,
                             max_queue_size=10,  callbacks=[history])

            # Save the max model, if any            
            if (history.history['val_acc'][0]>max_val):
                max_val= history.history['val_acc'][0]
                save_model(model=model,is_max_val_inclluded=True,max_val=max_val,ep=ep)
             
            # Save/overwrite the model
            save_model(model)
               
            ep += 1
            save_training_dynamics(ep,history,with_header=False)        

            # stop the traning if certain accuracy is reached
            #if (ep%10==0):
                #manual_testing()   
            #if   (history.history['val_acc'][0]>0.92):
              #  break
            if (history.history['loss'][0]<ideal_loss):
                   break
            
            if (ep > max_epoch):
                break

        return history
    
    # Transfer learning: freeze top layers but unfreeze all layers below the given layer   
def FreezeLayers(model, top_unfrozen_layer_name):
    
    model.trainable=True
    set_trainable = False
    for layer in model.layers:
        # Increase dropout rate
        if "dropout" in layer.name:
            layer.rate=0.7
            print (layer.name,"dropout rate updated to",layer.rate)
        if (layer.name==top_unfrozen_layer_name):
            set_trainable=True

        if (set_trainable):
            layer.trainable=True
        else:
            layer.trainable=False
        #if (layer.name=="dense_1"):
            #layer.trainable = False
    #model = add_new_dense(model)
    model_compile(model)
    return model

# add a new dense layer
def add_new_dense(model):
    new_model=Sequential()

    for layer in model.layers[:-1]:
        layer.name=layer.name+"_old"
        new_model.add(layer)
    new_model.add(Dense(units = 1024, activation='relu' ))
    new_model.add(Dropout(0.5))
    new_model.add(Dense (units=vocab_size, activation='softmax' ))
    return new_model

def PringFrozenLayers(model):
     for layer in model.layers:
            print ("Layer:",layer.name, "Frozen:",not layer.trainable)
            
def training_restart_initalize():
    import shutil
    shutil.copyfile(SETTINGS_DIR+"/Models/cnn_control.json", dnn_file_name_structure)
    shutil.copyfile(SETTINGS_DIR+"/Models/cnn_weight_control.h5", dnn_file_name_weights)
    if (os.path.isfile(training_dynamics_path)):
        os.remove(training_dynamics_path)
    print ("Ready for training...")

# Load X and y
training_set, test_set =get_train_test_sets()







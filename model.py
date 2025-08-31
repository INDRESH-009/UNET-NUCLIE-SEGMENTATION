import tensorflow as tf
import numpy as np 
import os
from tqdm import tqdm
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# input dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# data path 
TRAIN_PATH = 'data-science-bowl-2018/stage1_train'
TEST_PATH = 'data-science-bowl-2018/stage1_test'

# walk through the data folders to get the image ids 
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# creating a empty dataloader , we are going to fill it while resizing the all images to 128*128*3
X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
X_test = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)

print("Resizing training images and masks")
for n,id_ in tqdm(enumerate(train_ids),total =len(train_ids)):
    path = TRAIN_PATH+'/'+id_
    img = imread(path+'/images/'+id_+'.png')[:,:,:IMG_CHANNELS]
    img_resize = resize(img,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    # fill empty dataloader with the resizd image value 
    X_train[n] = img_resize
    
    # we have multiple mask files for the same image , so we are going to unify it by taking the max pixel count on all images so that we can make a unified mask 
    mask = np.zeros((IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
    for maskfile in next(os.walk(path+'/masks'))[2]:
        mask_ = imread(path+'/masks/'+maskfile)
        mask_resize = np.expand_dims(resize(mask_,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True),axis=-1)
        mask_unified = np.maximum(mask,mask_resize)
        
    # fill empty dataloader with the resizd unified mask value 
    Y_train[n] = mask_unified   
    
# dataloader for test images 
sizes_test = []
print("Resizing test images")
for n,id_ in tqdm(enumerate(test_ids),total=len(test_ids)):
    path = TEST_PATH+'/'+id_
    img = imread(path+'/images/'+id_+'.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0],img.shape[1]])
    img_resize = resize(img,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    # fill empty dataloader with the resizd image value 
    X_test[n] = img_resize
    
print('Data preparation completed')
    
# UNET IMPLEMENTATION

input_layer = tf.keras.layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
# normalise pixels btw [0,1]
x = tf.keras.layers.Lambda(lambda x:x/255)(input_layer)

# encoder/contraction path 
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(x)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c5)

# decoder structure 
u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c9)

# encoder decoder completed , now implementing the output by reducing the feature channels to 1 
output_layer = tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)

# model 
model = tf.keras.Model(inputs = [input_layer],outputs = [output_layer])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

############################
# Model checkpoint 
checkpointers = tf.keras.callbacks.ModelCheckpoint('model_for_nuclie_segmentation.h5',verbose=1,save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

callbacks = [checkpointers,early_stopping,tensorboard]

results = model.fit(X_train,Y_train,validation_split=0.1,batch_size=8,epochs=10,callbacks=callbacks)
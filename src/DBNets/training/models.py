#imports
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import RandomTranslation
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import numpy as np
import pickle

class CustomLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    o_mean = y_pred[:, :1]
    o_std = y_pred[:, 1:]
    loss = tf.reduce_mean(tf.math.log(o_std+1e-6) + tf.math.square((o_mean-y_true)/(o_std+1e-6))/2)
    return loss
    
class CustomLossWarmup(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    o_mean = y_pred[:, :1]
    _ = y_pred[:, 1:]
    loss = tf.reduce_mean(tf.math.square((o_mean-y_true)))
    return loss

########################## model with multiple parameters #############################################
def venus_multip(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    # augmentation and standardization
    x = RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest')(inputs)
    x = LayerNormalization(axis=[1,2])(x)
    x = GaussianNoise(0.01)(x)

    #first block
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #second block
    x = Conv2D(32, (3,3), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer )(x)
    x = Conv2D(32, (3,3), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #third block
    x = Conv2D(64, (3,3), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(64, (3,3), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)
 
    x = LayerNormalization(axis=[1,2])(x)
    
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x = Dense(256, activation=act, name='FC2', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)


    outLayer = Dense(6, activation='linear', name='o_mean')(x)
    simplecnn = Model(inputs, outLayer)

    return simplecnn
  
  
##########################model for ensemble###########################################################

def venus(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    # augmentation and standardization
    x = RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest')(inputs)
    x = LayerNormalization(axis=[1,2])(x)
    x = GaussianNoise(0.01)(x)

    #first block
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #second block
    x = Conv2D(32, (3,3), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer )(x)
    x = Conv2D(32, (3,3), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #third block
    x = Conv2D(64, (3,3), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(64, (3,3), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)
 
    x = LayerNormalization(axis=[1,2])(x)
    
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name='FC2', kernel_initializer=initializer)(x)
    
    #output
    mean = Dense(1, activation='linear', name='o_mean')(x1)
    std = Dense(1, activation='softplus', name='o_std')(x1)

    outLayer = Concatenate(axis=-1)([mean,std])
    
    simplecnn = Model(inputs, outLayer)

    return simplecnn


def venus_RI(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    # augmentation and standardization
    x = RandomFlip(mode="horizontal_and_vertical", seed=8365)(inputs)
    x = RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest')(x)
    x = LayerNormalization(axis=[1,2])(x)
    x = GaussianNoise(0.01)(x)

    #first block
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #second block
    x = Conv2D(32, (3,3), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer )(x)
    x = Conv2D(32, (3,3), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    x = LayerNormalization(axis=[1,2])(x)

    #third block
    x = Conv2D(64, (3,3), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(64, (3,3), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)
 
    x = LayerNormalization(axis=[1,2])(x)
    
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name='FC2', kernel_initializer=initializer)(x)
    
    #output
    mean = Dense(1, activation='linear', name='o_mean')(x1)
    std = Dense(1, activation='softplus', name='o_std')(x1)

    outLayer = Concatenate(axis=-1)([mean,std])
    
    simplecnn = Model(inputs, outLayer)

    return simplecnn


# OLD MODELS
######################################################################################################


def simpleCNNrotinvPc(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    #normalization
    x = LayerNormalization(axis=[1,2])(inputs)
    
    #noise
    #adding random noise
    x = GaussianNoise(noise, seed=(seed+383392)%49)(x)

    #x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    #y = x
    #first block
    x = Conv2D(8, (2,2), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(8, (2,2), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[1])(x)
    #x = Add()([x,y])

    #second block
    x = Conv2D(16, (5,5), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (5,5), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[2])(x)

    #third block
    x = Conv2D(32, (6,6), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(32, (6,6), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)
    #x = BatchNormalization()(x)
    
   # x = CyclPoolLayer()(x)
    
    #dense layers
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name='FC2', kernel_initializer=initializer)(x)
    #x1 = Dropout(dropout[5])(x1)

    #x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    #x2 = Dropout(dropout[5])(x2)
    
    #output
    mean = Dense(1, activation='linear', name='o_mean')(x1)
    std = Dense(1, activation='softplus', name='o_std')(x1)

    outLayer = Concatenate(axis=-1)([mean,std])
    #outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn

############################
def mercury(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    #normalization
    x = LayerNormalization(axis=[1,2])(inputs)
    
    #noise
    #adding random noise
    x = GaussianNoise(noise, seed=(seed+383392)%49)(x)

    #x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    #y = x
    #first block
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (3,3), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[1])(x)
    #x = Add()([x,y])

    #second block
    x = Conv2D(32, (5,5), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer)(x)
    x = Conv2D(32, (5,5), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[2])(x)

    #third block
    x = Conv2D(64, (3,3), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(64, (3,3), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)
    #x = BatchNormalization()(x)
    
   # x = CyclPoolLayer()(x)
    
    #dense layers
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name='FC2', kernel_initializer=initializer)(x)
    #x1 = Dropout(dropout[5])(x1)

    #x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    #x2 = Dropout(dropout[5])(x2)
    
    #output
    mean = Dense(1, activation='linear', name='o_mean')(x1)
    std = Dense(1, activation='softplus', name='o_std')(x1)

    outLayer = Concatenate(axis=-1)([mean,std])
    #outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn
###################

def ResNet3(input_shape=(64,64,1), act='leaky_relu', dropout=(0, 0, 0, 0.3, 0.3, 0.3), seed=0, noise=0):

    #initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    #define inputs
    inputs = Input(shape=input_shape)

    #normalization
    x = LayerNormalization(axis=[1,2])(inputs)
    
    #noise
    #adding random noise
    x = GaussianNoise(noise, seed=(seed+383392)%49)(x)

    #x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    x = Conv2D(8, (2,2), activation=act, padding='same', name='b1_c0', kernel_initializer=initializer)(x)
    y = x
    #first block
    x = Conv2D(8, (2,2), activation=act, padding='same', name='b1_c1', kernel_initializer=initializer)(x)
    x = Conv2D(8, (2,2), activation=act, padding='same', name='b1_c2', kernel_initializer=initializer)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[1])(x)
    x = Add()([x,y])

    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    x = Conv2D(16, (5,5), activation=act, padding='same', name='b2_c0', kernel_initializer=initializer)(x)
    y = x
    #second block
    x = Conv2D(16, (5,5), activation=act, padding='same', name='b2_c1', kernel_initializer=initializer)(x)
    x = Conv2D(16, (5,5), padding='same', activation=act, name='b2_c2', kernel_initializer=initializer)(x)
    x = Add()([x,y])
    #x = BatchNormalization()(x)
    #x = Dropout(dropout[2])(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    x = Conv2D(32, (6,6), activation=act, padding='same', name='b3_c0', kernel_initializer=initializer)(x)
    y = x
    #third block
    x = Conv2D(32, (6,6), activation=act, padding='same', name='b3_c1', kernel_initializer=initializer)(x)
    x = Conv2D(32, (6,6), padding='same', activation=act, name='b3_c2', kernel_initializer=initializer)(x)
    x = Add()([x,y])
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)

    #dense layers
    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name='FC2', kernel_initializer=initializer)(x)
    #x1 = Dropout(dropout[5])(x1)

    #x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    #x2 = Dropout(dropout[5])(x2)
    
    #output
    mean = Dense(1, activation='linear', name='o_mean')(x1)
    std = Dense(1, activation='softplus', name='o_std')(x1)

    outLayer = Concatenate(axis=-1)([mean,std])
    #outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn
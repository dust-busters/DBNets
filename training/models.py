#imports
import keras_cv
from keras.layers import LayerNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import RandomTranslation
from keras.layers import RandomFlip
from keras.layers import Concatenate
from keras.layers import GaussianNoise
from keras.layers import Add
from keras.layers import LeakyReLU
from keras_cv.layers import RandomAugmentationPipeline
from keras_cv.core import UniformFactorSampler
from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy as np
import keras
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

def get_gaussian_beam(size, sigma):
  print(sigma)
  xy = tf.linspace(-size/2, size/2, size)
  xx, yy = tf.meshgrid(xy, xy)
  filter = 1/(2.0*np.pi*sigma**2.0) * tf.math.exp(-(xx**2 + yy**2)/(2.0 * sigma**2.0))
  return filter


class RandomBeam(keras_cv.layers.BaseImageAugmentationLayer):
  
  def __init__(self, factor, seed=0, **kwargs):
    super().__init__(**kwargs)
    print(factor)
    self.factor = UniformFactorSampler(lower=0., upper=factor, seed=seed)
    print(self.factor.get_config())
    self.kernel_size = 10
    
    
  def augment_image(self, image, *args, transformation=None, **kwargs):
    kern = array_ops.expand_dims(array_ops.expand_dims(get_gaussian_beam(self.kernel_size, self.factor()), 2), 3)
    image =   array_ops.expand_dims(image, 0)
    ret = tf.nn.conv2d(image, kern, strides=(1,1), padding='VALID')
    return ret[0]
  
  
  def get_random_transformation(self, **kwargs):
        # kwargs holds {"images": image, "labels": label, etc...}
        return self.factor()
    
    
    
def venus_multip(input_shape=(64,64,1), act='leaky_relu', dropout=0.2, seed=0, noise=0, maximum_res=0.2):

  #define the augmentation pipeline
  augm_layers = [
                  RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest'),
                  GaussianNoise(noise),
                  RandomBeam(maximum_res*input_shape[0]/8.)
                 ]
  
  #initializer
  initializer = keras.initializers.HeNormal(seed=seed)

  #define inputs
  inputs = Input(shape=input_shape)
  
  #x = RandomAugmentationPipeline(augm_layers, augmentations_per_image=3, rate=0.5)(inputs)
  # augmentation and standardization
  
  x = LayerNormalization(axis=[1,2])(inputs)
  
  for l in augm_layers:
    x = l(x)
  

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
  x = Dropout(dropout)(x)
  x = Dense(256, activation=act, name='FC1', kernel_initializer=initializer)(x)
  x = Dropout(dropout)(x)
  x = Dense(256, activation=act, name='FC2', kernel_initializer=initializer)(x)
  x = Dropout(dropout)(x)
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
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from sklearn.model_selection import train_test_split

# To be able to run tests locally - replace with the local path to dataset
ship_dir = '/kaggle/input/airbus-ship-detection'
train_image_dir = os.path.join(ship_dir, 'train_v2')

# Data Preparation 
masks = pd.read_csv(os.path.join(ship_dir + '/train_ship_segmentations_v2.csv'))
not_empty = pd.notna(masks.EncodedPixels)
print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
masks.drop(['ships'], axis=1, inplace=True)
print(unique_img_ids.loc[unique_img_ids.ships>=2].head())

SAMPLES_PER_GROUP = 10
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)
print(balanced_train_df.shape[0], 'masks')

train_ids, valid_ids = train_test_split(balanced_train_df, 
                 test_size = 0.2, 
                 stratify = balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


# Function to load images from the directory
def load_images(image):
    """
    Load image data from the specified directory and preprocess it.
    
    Args:
    - image: Name of the image file
    
    Returns:
    - Preprocessed image data
    """
    path = train_image_dir + '/' + image
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    return image


# Function to decode RLE-encoded masks
def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decodes run-length encoded masks into numpy arrays representing masks.

    Args:
    - mask_rle: Run-length encoded mask as a string (start length)
    - shape: Tuple defining the (height, width) of the mask array to return

    Returns:
    - Numpy array representing the mask (1 for mask, 0 for background)
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

images_pixels = []
masks_pixels = []

images = train_df.groupby('ImageId')
for image_id, group_data in images:
    pixels = np.zeros((768, 768))
    for index, value in group_data['EncodedPixels'].items():
        if isinstance(value, str):
            pixels += rle_decode(value)
            
    images_pixels.append(load_images(image_id))
    masks_pixels.append(pixels)



# Model Building

# Function to define the input layer of the U-Net-like model
def input_layer():
    """
    Creates the input layer for the U-Net-like model.

    Returns:
    - Keras Input layer with a specific shape
    """
    return tf.keras.layers.Input(shape=(768, 768) + (3,))


# Function to create a downsample block in the U-Net-like model
def downsample_block(filters, size, batch_norm=True):
    """
    Creates a downsample block for the U-Net-like model.

    Args:
    - filters: Number of filters for the Convolutional layers
    - size: Size of the convolutional kernel
    - batch_norm: Boolean indicating whether to include batch normalization

    Returns:
    - Sequential model defining the downsample block
    """
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()
    
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    return result


# Function to create an upsample block in the U-Net-like model
def upsample_block(filters, size, dropout=False):
    """
    Creates an upsample block for the U-Net-like model.

    Args:
    - filters: Number of filters for the Transposed Convolutional layers
    - size: Size of the transposed convolutional kernel
    - dropout: Boolean indicating whether to include dropout

    Returns:
    - Sequential model defining the upsample block
    """
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()
    
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    
    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))
    
    result.add(tf.keras.layers.ReLU())
    return result


# Function to define the output layer of the U-Net-like model
def output_layer(size):
    """
    Creates the output layer for the U-Net-like model.

    Args:
    - size: Size of the transposed convolutional kernel

    Returns:
    - Keras Conv2DTranspose layer for the output
    """
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(1, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')

inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

x = inp_layer

downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)
    
downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

tf.keras.utils.plot_model(unet_like, show_shapes=True, dpi=72)



# Creating metric and loss
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Computes the Jaccard distance loss between predicted and true masks.

    Args:
    - y_true: Ground truth masks
    - y_pred: Predicted masks
    - smooth: Smoothing parameter to avoid division by zero

    Returns:
    - Jaccard distance loss value
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss

def dice_coef(y_true, y_pred, smooth=1):
    """
    Computes the Dice coefficient between predicted and true masks.

    Args:
    - y_true: Ground truth masks
    - y_pred: Predicted masks
    - smooth: Smoothing parameter to avoid division by zero

    Returns:
    - Dice coefficient value
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



# Model Compilation, Training  and Saving
unet_like.compile(optimizer='adam', loss=[jaccard_distance_loss], metrics=[dice_coef])

unet_like.fit(tf.convert_to_tensor(images_pixels), tf.convert_to_tensor(masks_pixels), epochs=10, batch_size=16)

unet_like.save('unet_like.keras')
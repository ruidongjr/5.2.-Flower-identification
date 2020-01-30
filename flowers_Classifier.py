"""
5. Machine learning
    5.2. Flower identification

    This script adopt CNN algorithm.

    step 1: Unzip the digits.zip file
    step 2: Re-arrange images into train+val+test
    step 3: Load train, test, and validation data sets
    step 4: Prepare data for CNN
    step 5: Build CNN
            Due to RAM limitation, only simplified model is build. Another complicated model topology is commented off.
    step 6: Testing

    Estimated Running Time: 10978s
    Test Accuracy: 71.4%

    Future works:
        1. With aid of cloud platform, complicated CNN model can be explored.
        2. DNN has been supported by universal approximation theorem that any continuous function can be approximated
    by a neural network with finite neurons. Fine tuning is needed to find the suitable hyper-parameters.
        3. Due to the difficulty of training a complicated CNN, deep transfer learning can be used, which is also proved
    by academia, to quicken the training time and boost the accuracy.

    Note:
        a. Run this script require large RAM, recommend to execute it in a cloud platform.
            For simplified CNN model, free platforms Kaggle/Colab are able to run.
            For complicated CNN model, need to go to AWS/Azure cloud.
        b. To reduce RAM space, do not read images before train_test_split, please make directories as follow:
            flowers_data
            │
            └───train
            |    │
            |    └───Daisy
            |    │
            |    └───Dandelion
            |    |
            |    └───Rose
            |    │
            |    └───Sunflower
            |    |
            |    └───Tulip
            └───valid
            |    │
            |    └───Daisy
            |    │
            |    └───Dandelion
            |    |
            |    └───Rose
            |    │
            |    └───Sunflower
            |    |
            |    └───Tulip
            └───test
                 │
                 └───Daisy
                 │
                 └───Dandelion
                 |
                 └───Rose
                 │
                 └───Sunflower
                 |
                 └───Tulip

"""

import numpy as np
from zipfile import ZipFile
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, Dense, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import tensorflow.keras.utils as utils
import os
from shutil import copyfile
from PIL import ImageFile
import matplotlib.pyplot as plt



def unzip(path='flowers.zip'):
    with ZipFile(path, 'r') as zip_parent:
        # extracting all the files
        print('Extracting all the files now...')
        zip_parent.extractall()
        print('Done')


def train_valid_test(files):
    """
    Within each category, split the files in training, validation and testing sets by indexing
    """
    x_idx, y_idx = np.arange(0, len(files), dtype=np.int), np.arange(0, len(files), dtype=np.int)
    # first split training + testing from all dataset
    X_t_idx, X_test_idx, y_t_idx, y_test_idx = train_test_split(x_idx, y_idx, test_size=0.2, random_state=1)
    # second split training + validating from training set
    X_train_idx, X_val_idx, y_train_idx, y_val_idx = train_test_split(X_t_idx, y_t_idx, test_size=0.2, random_state=1)

    train_files = [files[j] for j in X_train_idx]
    valid_files = [files[j] for j in X_val_idx]
    test_files = [files[j] for j in X_test_idx]
    return train_files, valid_files, test_files


def _copy_files(files, src, dest):
    """
    This function is a helper function to copy files from src to dest
    """
    for file in files:
        copyfile("{}/{}".format(src, file), "{}/{}".format(dest, file))


def copy_files(base_dir, categories):
    """
    This function copies files for all categories and data sets
    """
    total_images = []
    for category in categories:
        images = os.listdir("{}/{}".format(base_dir, category))
        # delete none .jpg files
        filtered_images = [image for image in images if image not in ['flickr.py', 'flickr.pyc', 'run_me.py']]

        total_images.append(len(filtered_images))

        train_images, valid_images, test_images = train_valid_test(filtered_images)

        _copy_files(train_images, "{}/{}".format(base_dir, category), "./flowers_data/train/{}".format(category))
        _copy_files(valid_images, "{}/{}".format(base_dir, category), "./flowers_data/valid/{}".format(category))
        _copy_files(test_images, "{}/{}".format(base_dir, category), "./flowers_data/test/{}".format(category))
    return total_images


def load_dataset(path):
    """
    Load image names + label image with category
    """
    data = load_files(path)
    flower_files = np.array(data['filenames'])
    # convert label to category
    flower_targets = utils.to_categorical(np.array(data['target']), 5)
    return flower_files, flower_targets


def _filename_to_data(img_path):
    """
    Helper function for paths_to_tensor(img_paths)
    unify the image size
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with unified shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def filename_to_data(img_paths):
    """
    This function read file names into data
    """
    list_of_tensors = [_filename_to_data(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


class CNN:
    '''
          deep neural network (CNN)
          arg:  input_size = input array size
                lr = learning rate
                decay
                loss = cost function
                metrics = evaluation metrics
                batch_size = mini-batch process
                epochs = training times

    '''

    def __init__(self, input_size, learning_rate, decay, loss, metrics, batch_size, epochs):
        # initialize hyper-parameters, parameters of CNN
        self.input_size = input_size
        self.lr = learning_rate
        self.decay = decay
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = Sequential()
        self.opt_rms = optimizers.RMSprop(lr=self.lr, decay=self.decay)

    # Plot function
    def plot_history(self, model_history):
        plt.plot(model_history.history['acc'])
        plt.plot(model_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    # Helper function to create CNN block
    def addBlock(self, filters, dropout):
        self.model.add(Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001),
                         input_shape=self.input_size))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(dropout))

    # function to plot confusion matrix
    def plot_confusion_matrix(self, df_confusion, title='Confusion matrix', cmap=plt.cm.gist_gray_r):
        plt.matshow(df_confusion, cmap=cmap)  # imshow
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(df_confusion.columns))
        plt.xticks(tick_marks, df_confusion.columns, rotation=45)
        plt.yticks(tick_marks, df_confusion.index)
        # plt.tight_layout()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)

    def fit(self, X, y, X_val, y_val):
        self.model.compile(loss=self.loss, optimizer=self.opt_rms, metrics=self.metrics)
        cnn_history = self.model.fit_generator(datagen.flow(X, y, batch_size=self.batch_size),
                                          steps_per_epoch=X.shape[0] // self.batch_size, epochs=self.epochs,
                                          verbose=1, validation_data=(X_val, y_val))
        return cnn_history

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.batch_size, verbose=1)


if __name__ == '__main__':
    # 1. Unzip the digits.zip file
    unzip()

    # 2. re-arrange data into train+val+test
    base_dir = "./flowers/"
    categories = os.listdir(base_dir)
    total_images = copy_files(base_dir, categories)

    # 3. load train, test, and validation datasets
    train_files, train_label = load_dataset('flowers_data/train')
    valid_files, valid_label = load_dataset('flowers_data/valid')
    test_files, test_label = load_dataset('flowers_data/test')

    # 4. prepare data for CNN
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train = filename_to_data(train_files).astype('float32')
    valid = filename_to_data(valid_files).astype('float32')
    test = filename_to_data(test_files).astype('float32')

    # center and normalise data
    mean = np.mean(train)
    std = np.std(train)
    train = (train - mean) / (std)
    valid = (valid - mean) / (std)
    test = (test - mean) / (std)

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(train)

    # 5. build CNN
    my_CNN = CNN(input_size=train.shape[1:],
                 learning_rate=0.001,
                 decay=1e-6,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 batch_size=64,
                 epochs = 120)


    # complicated model: if large RAM is available, this model will produce higher accuracy
    # my_CNN.addBlock(128, 0.2)
    # my_CNN.addBlock(256, 0.3)
    # my_CNN.addBlock(512, 0.4)
    # my_CNN.model.add(Flatten())
    # my_CNN.model.add(Dense(5, activation='softmax'))
    # my_CNN.model.summary()


    # simplified model: suitable for low RAM.
    my_CNN.model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224, 224, 3)))
    my_CNN.model.add(MaxPooling2D(pool_size=2, strides=2))
    my_CNN.model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation='relu'))
    my_CNN.model.add(MaxPooling2D(pool_size=2, strides=2))
    my_CNN.model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
    my_CNN.model.add(MaxPooling2D(pool_size=2, strides=2))
    my_CNN.model.add(GlobalAveragePooling2D())
    my_CNN.model.add(Dense(5, activation='softmax'))
    my_CNN.model.summary()


    # training
    cnn_history = my_CNN.fit(train, train_label, valid, valid_label)

    # plot learning curves
    my_CNN.plot_history(cnn_history)

    # 6. testing
    scores = my_CNN.evaluate(test, test_label)
    print('\nTest accuracy: %.3f ' % scores[1])

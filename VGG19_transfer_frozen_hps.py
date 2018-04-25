# Here, we will be making an attempt to perform transfer learning with the SYSU nighttime dataset
# and we will be doing it in a (potentially naive) binary classifier of Vehicular VS. Non-Vehicular
# classes. We will only be training some fully connected layers at the end, no conv kernel fine tuning
#
from time import time
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, metrics
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, TensorBoard

import os, os.path
import sys

from sacred import Experiment
from sacred.observers import MongoObserver
from labwatch.assistant import LabAssistant
from labwatch.optimizers.random_search import RandomSearch
from labwatch.hyperparameters import UniformNumber, UniformFloat


ex = Experiment()
a = LabAssistant(ex, optimizer=RandomSearch)

@a.search_space
def search_space():
    num_units_first_dense = UniformNumber(lower=950,
                                          upper=1450,
                                          default=1024,
                                          type=int,
                                          log_scale=True)
    num_units_second_dense = UniformNumber(lower=900,
                                           upper=1600,
                                           default=1024,
                                           type=int,
                                           log_scale=True)
    dropout_rate = UniformFloat(lower=.33,
                                upper=.55,
                                default=.45)
    lr = UniformFloat(lower=.003,
                      upper=.007,
                      default=.004)
    momentum = UniformFloat(lower=.4,
                            upper=.93,
                            default=.5)

# Defaults
@ex.config
def config():
    num_units_first_dense = 1000
    dropout_rate = 0.5
    num_units_second_dense = 1000
    lr = .001
    momentum = .001


@ex.automain
def run(num_units_first_dense,
        dropout_rate,
        num_units_second_dense,
        lr,
        momentum):

    img_width, img_height = 128,128
    train_dir = './cropped_split_dataset/train'
    validation_dir = './cropped_split_dataset/validation'


    model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in model.layers:
        layer.trainable = False

    l = model.output
    l = Flatten()(l)
    l = Dense(num_units_first_dense, activation='relu')(l)
    l = Dropout(dropout_rate)(l)
    l = Dense(num_units_second_dense, activation='relu')(l)
    final = Dense(2, activation='softmax')(l)
    final_model = Model(inputs=model.input, outputs=final)
    final_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=lr, momentum=momentum),
        metrics=['accuracy'])

    # Image Data Generator/Augmentator values!
    #    - batch_size:  accepted
    #    - all data augmentation constants
    prod_aug = {
        'horizontal_flip': True,
        'zoom_range' : 0.3,
        'width_shift_range': 0.3,
        'height_shift_range': 0.3,
        'rotation_range': 10}

    batch_size = 32
    epochs = 2

    # NO AUGMENTATION APPLIED FOR HYPERPARAM SEARCH
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=prod_aug['horizontal_flip'],
                                       zoom_range=prod_aug['zoom_range'],
                                       width_shift_range=prod_aug['width_shift_range'],
                                       height_shift_range=prod_aug['height_shift_range'],
                                       rotation_range=prod_aug['rotation_range'])
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=batch_size,
        target_size=(img_height, img_width),
        class_mode='categorical')

    # Save model weights and save model to json
    # weight_file = 'VGG19_transfer_1.h5'
    # json_file = 'VGG19_tansfer_1.json'
    # checkpoint = ModelCheckpoint(filepath='./weights/' + weight_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # model_json = model.to_json()
    # with open('./models_json/' + json_file, 'w') as json_file:
    #    json_file.write(model_json)



    early = EarlyStopping(monitor='val_acc', min_delta=.3, patience=50, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    num_train_ex = len([file for file in os.listdir(train_dir + '/cars/')]) + len([file for file in os.listdir(train_dir + '/non-cars/')])
    num_validation_ex = len([file for file in os.listdir(validation_dir + '/cars/')]) + len([file for file in os.listdir(validation_dir + '/non-cars/')])

    hist = final_model.fit_generator(
                    train_generator,
                    steps_per_epoch=int(num_train_ex/batch_size),
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=int(num_validation_ex/batch_size),
                    use_multiprocessing=True,
                    callbacks=[early, tensorboard])

    return hist.history['val_acc'][0]

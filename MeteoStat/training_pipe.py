import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import os

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox


# Collect all filenames
def collect_filenames():
    """
    This function returns a list of all the filenames in a folder
    The returned list is sorted (ascending)
    """
    filenames = []
    for filepath in os.listdir('/content/drive/MyDrive/Projet_MeteoStat/images_preproc/'):
        filenames.append(filepath)
    return sorted(filenames)

# Panda DF Creation
def create_df_all_images(sorted_filenames):
    """
    Input : List
    Returns : DataFrame

    This function returns a dataframe with one row for each filename from input variable
    Row order is the same as input list order
    The DF will have Name, NameYear, Month, Day, Time and FileFormat
    """
    series_name = pd.Series(sorted_filenames)
    df_all = pd.DataFrame(series_name, columns=["Filename"])
    df_all[['Name','NameYear','Month', 'Day', 'Time']] = df_all['Filename'].str.split('_',expand=True)
    df_all[['Time', 'FileFormat']] = df_all['Time'].str.split('.', expand=True)
    # df_all.groupby(['Month', 'Day']).count()
    return df_all

# Split DataFrame into train and test
def trainer_test_split(df, frac=0.2):

    """
    Input : DataFrame and optional frac of how much percentage should be test set
            frac default value = 0.2
    Returns : DataFrame train and DataFrame test

    The split works so that the test rows are taken from the tail of the dataframe
    """
    test = df.tail(int(len(df.index)*frac))
    train = df.drop(index=test.index)

    return train, test


def generate_one_set_names(first_index,df):
    """
    Input : index value to start the set, and dataframe
    Returns : One set of 20 sorted filenames ordered from the index value and forward

    The split works so that the test rows are taken from the tail of the dataframe
    """

    sorted_filenames = []
    for i in range(first_index,first_index+20):
      sorted_filenames.append(pd.Series(df['Filename'])[i])

    return sorted_filenames

def generate_all_set_names(number_of_sets,df):
    """
    Input : Number of sets to generate (INT) and Dataframe
    Returns : a long list of all filenames ordered set by set

    How it works, random variable that selects index in DF, and sends the index to
    generate_one_set_names function. Thereafter it keeps the returned list in a
    large list
    """


    all_filenames = []
    min_index = min(list(df.index))
    max_index = max(list(df.index))-20
    for i in range(number_of_sets):
        num = random.randint(min_index,max_index)
        tmp = generate_one_set_names(num, df)
        all_filenames.extend(tmp)
    return all_filenames

def create_sets(sorted_filenames):

    """
    Input : A list of sorted filenames
    Returns : A numpy array of all the images corresponding to the filename
              Array output shape example: (30,20,95, 120) corresponding to 30 sets
              of 20 images with 95 x 120 pixels

    Currently the function generates gray scale out of colors and the folder where
    the images are found is hardcoded in the function
    """

    instances = []
    # instances = np.array(instances)
    temporary_list = []
    # temporary_list = np.array(temporary_list)
    # Load in the images
    counter = 0

    for name in sorted_filenames:

        tmp = cv2.imread('/content/drive/MyDrive/Projet_MeteoStat/images_preproc/{}'.format(name))
        tmp = tmp[::5,::5,:]
        gray_image = tmp.dot([0.07, 0.72, 0.21])
        # temporary_list = np.append(temporary_list,gray_image)
        temporary_list.append(gray_image)

        if counter>=19:
            # print(name)
            # instances = np.append(instances, temporary_list,axis=1)
            instances.append(temporary_list)
            counter=-1
            temporary_list = []
            # temporary_list = np.array(temporary_list)
        counter+=1


    return np.array(instances)


def arrange_training_data(test):
    # Swap the axes representing the number of frames and number of data samples.
    dataset = np.expand_dims(test, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.8 * dataset.shape[0])] # Was 0.9
    val_index = indexes[int(0.2 * dataset.shape[0]) :] # Was 0.1
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255
    return train_dataset,val_dataset


# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    X = data[:,  : 10, :, :]
    y = data[:, 10 : 20, :, :]
    return X, y

def visualize_one_set(train_dataset):

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one random data example.
    data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    # Print information and display the figure.
    print(f"Displaying frames for example {data_choice}.")
    plt.show()


def create_model(X_train):

    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *X_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(32, 32),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    # x = layers.BatchNormalization()(x)
    # x = layers.ConvLSTM2D(
    #     filters=8,
    #     kernel_size=(16, 16),
    #     padding="same",
    #     return_sequences=True,
    #     activation="relu",
    # )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(4, 4),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ConvLSTM2D(
    #     filters=64,
    #     kernel_size=(1, 1),
    #     padding="same",
    #     return_sequences=True,
    #     activation="relu",
    # )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="linear", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),) # was loss: binary_crossentropy

    model.summary()
    return model

def train_model(model, X_train, y_train,X_val,y_val,epochs=5,batch_size=5):

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


    # Fit the model to the training data.
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )
    return model

def visualize_one_pred(val_dataset, model):

    # Select a random example from the validation dataset.
    example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]


    # Predict a new set of 10 frames.
    for _ in range(10):
        # Extract the model's prediction and post-process it.
        new_prediction = model.predict(np.expand_dims(frames, axis=0))

        new_prediction = np.squeeze(new_prediction, axis=0)

        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)


        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame), axis=0)

    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    new_frames = frames[10:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Display the figure.
    plt.show()

def generate_gifs(val_dataset, model):

    # Select a few random examples from the dataset.
    examples = val_dataset[np.random.choice(range(len(val_dataset)), size=5)]

    # Iterate over the examples and predict the frames.
    predicted_videos = []
    for example in examples:
        # Pick the first/last ten frames from the example.
        frames = example[:10, ...]
        original_frames = example[10:, ...]
        new_predictions = np.zeros(shape=(10, *frames[0].shape))

        # Predict a new set of 10 frames.
        for i in range(10):
            # Extract the model's prediction and post-process it.
            frames = example[: 10 + i + 1, ...]
            new_prediction = model.predict(np.expand_dims(frames, axis=0))
            new_prediction = np.squeeze(new_prediction, axis=0)
            predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

            # AJ temporary white augmentation and black if below 0.25
            # predicted_frame = np.where(predicted_frame<0.001,0,predicted_frame*2)

            # Extend the set of prediction frames.
            new_predictions[i] = predicted_frame

        # Create and save GIFs for each of the ground truth/prediction images.
        for frame_set in [original_frames, new_predictions]:
            # Construct a GIF from the selected video frames.
            current_frames = np.squeeze(frame_set)
            current_frames = current_frames[..., np.newaxis] * np.ones(3)
            current_frames = (current_frames * 255).astype(np.uint8)
            current_frames = list(current_frames)

            # Construct a GIF from the frames.
            with io.BytesIO() as gif:
                imageio.mimsave(gif, current_frames, "GIF", fps=5)
                predicted_videos.append(gif.getvalue())

    # Display the videos.
    print(" Truth\tPrediction")
    for i in range(0, len(predicted_videos), 2):
        # Construct and display an `HBox` with the ground truth and prediction.
        box = HBox(
            [
                widgets.Image(value=predicted_videos[i]),
                widgets.Image(value=predicted_videos[i + 1]),
            ]
        )
        display(box)

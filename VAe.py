# Importing necessary packages
from __future__ import print_function

import argparse
import time
from time import asctime
import numpy as np
import multiprocessing as mp
from dataset import *
import numpy as np
import cv2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Input, Lambda, \
    Flatten, Reshape, Conv2DTranspose
import tensorflow.keras.backend as K
from random import randint as r
import glob
import gc
import psutil
import copy
import matplotlib.pyplot as plt


class VAE():
    def __init__(self):

        self.model_name = 'test_batchnorm'
        self.version = ""
        self.save_dir = self.model_name + "v" + self.version

        self.data_dir = r"W:\Projects\General\FDGAN\kiryatgat-1502-fdgan-master\CelebA\img_align_celeba\img_align_celeba"
        self.log_dir = self.save_dir + "/logs/"
        self.sample_dir = self.save_dir + '/samples/'
        self.test_dir = self.save_dir + '/test/'

        self.sample_size = 100000

        self.shape = None
        self.sd_layer = None
        self.mean_layer = None
        self.stride = 2

        # These are mean and standard deviation values obtained from the celebA dataset used for training
        self.mean = 0.43810788
        self.std = 0.29190385

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self.batch_size = 64
        self.epochs = 50
        self.input_size = 32
        self.input_shape = (self.input_size, self.input_size, 3)
        self.inp = Input(self.input_shape)

        self.data_set = CelebA(output_size=self.input_size, channel=self.input_shape[-1], sample_size=self.sample_size,
                               batch_size=self.batch_size, crop=True, filter=False, data_dir=self.data_dir,
                               ignore_image_description=True)

        # callbacks
        # self.early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
        # self.checkpoint_callback = ModelCheckpoint(save_dir + 'model_best_weights.h5', monitor='loss', verbose=1, save_best_only=True,
        #                              mode='min', period=1)
        # self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    def vae_loss(self, input_img, output):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(output - input_img))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + self.sd_layer - K.square(self.mean_layer) - K.square(K.exp(self.sd_layer)), axis=-1)
        # return the average loss over all images in batch
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss

    def sampler(self, layers):
        std_norm = K.random_normal(shape=(K.shape(layers[0])[0], 128), mean=0, stddev=1)
        return layers[0] + layers[1] * std_norm

    # Building the Encoder
    def build_encoder(self):
        x = self.inp
        x = Conv2D(32, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        self.shape = K.int_shape(x)
        x = Flatten()(x)

        x = Dense(256, activation="relu")(x)

        self.mean_layer = Dense(128, activation="relu")(x)  # should sigmoid
        self.mean_layer = BatchNormalization()(self.mean_layer)

        self.sd_layer = Dense(128, activation="relu")(x)  # should sigmoid
        self.sd_layer = BatchNormalization()(self.sd_layer)

        # latent_vector = Lambda(self.sampler)([self.mean_layer, self.sd_layer])
        return Model(self.inp, [self.mean_layer, self.sd_layer], name="VAE_Encoder")

    # Building the decoder
    def build_decoder(self):
        decoder_inp = Input(shape=(128,))
        x = decoder_inp
        x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation="relu")(x)

        x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)

        x = (Conv2DTranspose(32, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
        x = BatchNormalization()(x)

        x = (Conv2DTranspose(16, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
        x = BatchNormalization()(x)

        x = (Conv2DTranspose(8, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
        x = BatchNormalization()(x)

        outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
        # should RELU

        return Model(decoder_inp, outputs, name="VAE_Decoder")

    def build_autoencoder(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.autoencoder = Model(self.inp, self.decoder(self.encoder(self.inp)), name="Variational_Auto_Encoder")

        self.autoencoder.compile(optimizer="adam", loss=self.vae_loss, metrics=["accuracy"],
                                 experimental_run_tf_function=False)

        if os.path.exists(self.model_name + ".h5"):
            self.autoencoder.load_weights(self.model_name + ".h5")  # Loading pre-trained weights

        return self.autoencoder

    def train(self):

        imgs = glob.glob(self.data_dir + "/*.jpg")
        print('='*20)
        print("[+] found ", len(imgs), " images in database\nloading images...")

        train_y = []
        train_y2 = []

        load_time = time.Time()
        for _ in range(0, self.sample_size):
            if _ % 500 == 0:
                print("[{}%]  {} / {}".format(_//self.sample_size*100, _, self.sample_size), end="\r")
            img = cv2.imread(imgs[_])
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            train_y.append(img.astype("float32") / 255.0)

        print("[+] done loading trainY1 - took {:f.2} seconds".format(load_time - time.Time()))

        load_time = time.Time()
        for _ in range(self.sample_size, self.sample_size * 2):
            if _ % 500 == 0:
                print("[{}%]  {} / {}".format(_//(self.sample_size*2)*100, _, self.sample_size*2), end="\r")
            img = cv2.imread(imgs[_])
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            train_y2.append(img.astype("float32") / 255.0)

        print("[+] done loading trainY2 - took {:f.2} seconds".format(load_time - time.Time()))

        train_y = np.array(train_y)
        train_y2 = np.array(train_y2)
        Y_data = np.vstack((train_y, train_y2))
        del train_y, train_y2
        gc.collect()
        print("Virtual memory: ", psutil.virtual_memory())
        Z_data = copy.deepcopy(Y_data)
        Z_data = (Z_data - Z_data.mean()) / Z_data.std()
        print("===Starting Training===\n")
        self.autoencoder.fit(Z_data, Y_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=0)

        test_Y = []

        load_time = time.Time()
        for _ in range(200000, 202599):
            if _ % 500 == 0:
                print("{} / 100000".format(_), end='\r')
            img = cv2.imread(imgs[_])
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            test_Y.append(img.astype("float32") / 255.0)

        print("[+] done loading test - took {:f.2} seconds".format(load_time - time.Time()))

        test_Y = np.array(test_Y)
        mean = test_Y.mean()
        std = test_Y.std()
        test_Z = (test_Y - mean) / std

        pred = self.autoencoder.predict(test_Z)
        temp = r(0, 2599)
        print(temp)
        plt.subplot(1, 3, 1)
        plt.imshow(test_Y[temp])
        plt.subplot(1, 3, 2)
        plt.imshow(test_Z[temp])
        plt.subplot(1, 3, 3)
        plt.imshow(pred[temp])
        cv2.imshow("generated", pred[temp])

        # if os.path.exists()
        self.autoencoder.save_weights(self.model_name + ".h5")

    def generate(self, image=None):
        if image is None:
            img = np.random.normal(size=(9, 32, 32, 3))

            prediction = self.autoencoder.predict(img)
            op = np.vstack((np.hstack((prediction[0], prediction[1], prediction[2])),
                            np.hstack((prediction[3], prediction[4], prediction[5])),
                            np.hstack((prediction[6], prediction[7], prediction[8]))))
            print(op.shape)
            op = cv2.resize(op, (288, 288), interpolation=cv2.INTER_AREA)
            cv2.imshow("generated", op)
            cv2.imwrite("generated" + str(r(0, 9999)) + ".jpg", (op * 255).astype("uint8"))

        else:
            img = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            img = (img - self.mean) / self.std

            pred = self.autoencoder.predict(img.reshape(1, 32, 32, 3))
            cv2.imshow("prediction", cv2.resize(pred[0], (96, 96), interpolation=cv2.INTER_AREA))

        while cv2.waitKey(0) != 27:
            pass
        cv2.destroyAllWindows()


# ap = argparse.ArgumentParser()
# ap.add_argument("--weights", "-w", required=True)
# ap.add_argument("--generate", "-g", required=True, type=int,
#                 default=0)  # pass 1 as CommandLine argument to generate faces
# ap.add_argument("--image", "-i", required=False)
# args = vars(ap.parse_args())


def main():
    choice = input("what: ")
    vae = VAE()
    vae.build_autoencoder()
    if choice == 't':
        vae.train()
    elif choice == 'g':
        vae.generate()


if __name__ == "__main__":
    main()

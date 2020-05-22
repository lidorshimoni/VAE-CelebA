# Importing necessary packages
# from __future__ import print_function

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
    Flatten, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from random import randint as r
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import gc
import psutil
import copy
import matplotlib.pyplot as plt


class VAE:
    def __init__(self):

        self.model_name = 'anime'
        self.version = "1"
        self.save_dir = self.model_name + "v" + self.version + "/"

        self.data_dir = r"X:\Projects\2DO\anime-faces"
        # self.data_dir = r"W:\Projects\Done\FDGAN\kiryatgat-1502-fdgan-master\CelebA\img_align_celeba"
        self.log_dir = self.save_dir + "/logs/"
        self.sample_dir = self.save_dir + '/samples/'
        self.test_dir = self.save_dir + '/test/'

        self.sample_size = 5000

        self.shape = None
        self.sd_layer = None
        self.mean_layer = None
        self.shape_before_flattening = None
        self.decoder_output = None
        self.stride = 2

        # These are mean and standard deviation values obtained from the celebA dataset used for training
        self.mean = 0.43810788
        self.std = 0.29190385

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self.batch_size = 256
        self.epochs = 100
        self.input_size = 64
        self.encoder_output_dim = 256
        self.decoder_input = Input(shape=(self.encoder_output_dim,), name='decoder_input')
        self.input_shape = (self.input_size, self.input_size, 3)
        self.encoder_input = Input(shape=self.input_shape, name='encoder_input')

        # self.data_set = CelebA(output_size=self.input_size, channel=self.input_shape[-1], sample_size=self.sample_size,
        #                        batch_size=self.batch_size, crop=True, filter=False, data_dir=self.data_dir,
        #                        ignore_image_description=True)

        self.use_batch_norm = True
        self.use_dropout = False
        self.LOSS_FACTOR = 10000
        self.learning_rate = 0.0005
        self.adam_optimizer = Adam(lr=self.learning_rate)

        # callbacks
        # checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/weights.h5'), save_weights_only=True,
        #                                  verbose=1)

        # self.early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
        self.checkpoint_callback = ModelCheckpoint(self.save_dir + self.model_name + '_best.h5', monitor='loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   mode='min', period=1)
        # self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # def vae_loss(self, input_img, output): # compute the average MSE error, then scale it up, ie. simply sum on all
    # axes reconstruction_loss = K.sum(K.square(output - input_img)) # compute the KL loss kl_loss = - 0.5 * K.sum(1
    # + self.sd_layer - K.square(self.mean_layer) - K.square(K.exp(self.sd_layer)), axis=-1) # return the average
    # loss over all images in batch total_loss = K.mean(reconstruction_loss + kl_loss) return total_loss

    def kl_loss(self, y_true, y_pred):
        kl_loss = -0.5 * K.sum(1 + self.sd_layer - K.square(self.mean_layer) - K.exp(self.sd_layer), axis=1)
        return kl_loss

    # def r_loss(self, y_true, y_pred):
    #     return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
    #

    # MSE loosssss
    def r_loss(self, y_true, y_pred):
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

    def total_loss(self, y_true, y_pred):
        # return self.LOSS_FACTOR * self.r_loss(y_true, y_pred) + self.kl_loss(y_true, y_pred)
        return K.mean(self.r_loss(y_true, y_pred) + self.kl_loss(y_true, y_pred))

    def sampler(self, layers):
        std_norm = K.random_normal(shape=(K.shape(layers[0])[0], 128), mean=0, stddev=1)
        return layers[0] + layers[1] * std_norm

    # Building the Encoder
    # def build_encoder(self):
    #     x = self.inp
    #     x = Conv2D(32, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
    #     x = BatchNormalization()(x)
    #
    #     x = Conv2D(64, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
    #     x = BatchNormalization()(x)
    #
    #     x = Conv2D(128, (2, 2), strides=self.stride, activation="relu", padding="same")(x)
    #     x = BatchNormalization()(x)
    #
    #     self.shape = K.int_shape(x)
    #     x = Flatten()(x)
    #
    #     x = Dense(256, activation="relu")(x)
    #
    #     self.mean_layer = Dense(128, activation="relu")(x)  # should sigmoid
    #     self.mean_layer = BatchNormalization()(self.mean_layer)
    #
    #     self.sd_layer = Dense(128, activation="relu")(x)  # should sigmoid
    #     self.sd_layer = BatchNormalization()(self.sd_layer)
    #
    #     # latent_vector = Lambda(self.sampler)([self.mean_layer, self.sd_layer])
    #     return Model(self.inp, [self.mean_layer, self.sd_layer], name="VAE_Encoder")
    #
    # # Building the decoder
    # def build_decoder(self):
    #     decoder_inp = Input(shape=(128,))
    #     x = decoder_inp
    #     x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation="relu")(x)
    #
    #     x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)
    #
    #     x = (Conv2DTranspose(32, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
    #     x = BatchNormalization()(x)
    #
    #     x = (Conv2DTranspose(16, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
    #     x = BatchNormalization()(x)
    #
    #     x = (Conv2DTranspose(8, (3, 3), strides=self.stride, activation="relu", padding="same"))(x)
    #     x = BatchNormalization()(x)
    #
    #     outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
    #     # should RELU
    #
    #     return Model(decoder_inp, outputs, name="VAE_Decoder")

    def build_encoder(self):

        conv_filters = [32, 64, 64, 64]
        conv_kernel_size = [3, 3, 3, 3]
        conv_strides = [2, 2, 2, 2]

        # Number of Conv layers
        n_layers = len(conv_filters)

        # Define model input
        x = self.encoder_input

        # Add convolutional layers
        for i in range(n_layers):
            x = Conv2D(filters=conv_filters[i],
                       kernel_size=conv_kernel_size[i],
                       strides=conv_strides[i],
                       padding='same',
                       name='encoder_conv_' + str(i)
                       )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        # Required for reshaping latent vector while building Decoder
        self.shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        self.mean_layer = Dense(self.encoder_output_dim, name='mu')(x)
        self.sd_layer = Dense(self.encoder_output_dim, name='log_var')(x)

        # Defining a function for sampling
        def sampling(args):
            mean_mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.)
            return mean_mu + K.exp(log_var / 2) * epsilon

        # Using a Keras Lambda Layer to include the sampling function as a layer
        # in the model
        encoder_output = Lambda(sampling, name='encoder_output')([self.mean_layer, self.sd_layer])

        return Model(self.encoder_input, encoder_output, name="VAE_Encoder")

    # Building the decoder
    def build_decoder(self):
        conv_filters = [64, 64, 32, 3]
        conv_kernel_size = [3, 3, 3, 3]
        conv_strides = [2, 2, 2, 2]

        n_layers = len(conv_filters)

        # Define model input
        decoder_input = self.decoder_input

        # To get an exact mirror image of the encoder
        x = Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = Reshape(self.shape_before_flattening)(x)

        # Add convolutional layers
        for i in range(n_layers):
            x = Conv2DTranspose(filters=conv_filters[i],
                                kernel_size=conv_kernel_size[i],
                                strides=conv_strides[i],
                                padding='same',
                                name='decoder_conv_' + str(i)
                                )(x)

            # Adding a sigmoid layer at the end to restrict the outputs
            # between 0 and 1
            if i < n_layers - 1:
                x = LeakyReLU()(x)
            else:
                x = Activation('sigmoid')(x)

        # Define model output
        self.decoder_output = x

        return Model(decoder_input, self.decoder_output, name="VAE_Decoder")

    def build_autoencoder(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Input to the combined model will be the input to the encoder.
        # Output of the combined model will be the output of the decoder.
        self.autoencoder = Model(self.encoder_input, self.decoder(self.encoder(self.encoder_input)),
                                 name="Variational_Auto_Encoder")

        self.autoencoder.compile(optimizer=self.adam_optimizer, loss=self.total_loss,
                                 metrics=[self.total_loss],
                                 # metrics=[self.r_loss, self.kl_loss],
                                 experimental_run_tf_function=False)
        self.autoencoder.summary()

        if os.path.exists(self.save_dir):
            if os.path.exists(self.save_dir + self.model_name + ".h5"):
                self.autoencoder.load_weights(self.save_dir + self.model_name + ".h5")  # Loading pre-trained weights
                print("===Loaded model weights===")
            if os.path.exists(self.save_dir + self.model_name + '_best.h5'):
                self.autoencoder.load_weights(self.save_dir + self.model_name + '_best.h5')
                print("===Loaded best model===")
        else:
            os.makedirs(self.save_dir)

        plot_model(
            self.autoencoder,
            to_file="model.png",
            show_shapes=True)
        return self.autoencoder

    def train(self):

        filenames = np.array(glob.glob(os.path.join(self.data_dir, '*/*.png')))
        # filenames = np.array(glob.glob(os.path.join(self.data_dir, '*/*.jpg')))
        NUM_IMAGES = len(filenames)
        print("Total number of images : " + str(NUM_IMAGES))

        data_flow = ImageDataGenerator(rescale=1. / 255).flow_from_directory(self.data_dir,
                                                                             target_size=self.input_shape[:2],
                                                                             batch_size=self.batch_size,
                                                                             shuffle=True,
                                                                             class_mode='input',
                                                                             subset='training'
                                                                             )

        self.autoencoder.fit_generator(data_flow,
                                       shuffle=True,
                                       epochs=self.epochs,
                                       initial_epoch=0,
                                       steps_per_epoch=NUM_IMAGES // (self.batch_size * 2),
                                       callbacks=[self.checkpoint_callback]
                                       )

        self.autoencoder.save_weights(self.save_dir + self.model_name + ".h5")

    # def train(self):
    #
    #     imgs = glob.glob(self.data_dir + "/*.jpg")
    #     print('='*20)
    #     print("[+] found ", len(imgs), " images in database\nloading images...")
    #
    #     train_y = []
    #     train_y2 = []
    #
    #     load_time = time.time()
    #     for _ in range(0, self.sample_size):
    #         if _ % 500 == 0:
    #             print("[{}%]  {} / {}".format(_//self.sample_size*100, _, self.sample_size), end="\r")
    #         img = cv2.imread(imgs[_])
    #         img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    #         train_y.append(img.astype("float32") / 255.0)
    #
    #     print("[+] done loading trainY1 - took {:.2f} seconds".format(load_time - time.time()))
    #
    #     load_time = time.time()
    #     for _ in range(self.sample_size, self.sample_size * 2):
    #         if _ % 500 == 0:
    #             print("[{}%]  {} / {}".format(_//(self.sample_size*2)*100, _, self.sample_size*2), end="\r")
    #         img = cv2.imread(imgs[_])
    #         img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    #         train_y2.append(img.astype("float32") / 255.0)
    #
    #     print("[+] done loading trainY2 - took {:.2f} seconds".format(load_time - time.time()))
    #
    #     train_y = np.array(train_y)
    #     train_y2 = np.array(train_y2)
    #     Y_data = np.vstack((train_y, train_y2))
    #     del train_y, train_y2
    #     gc.collect()
    #     print("Virtual memory: ", psutil.virtual_memory())
    #     Z_data = copy.deepcopy(Y_data)
    #     Z_data = (Z_data - Z_data.mean()) / Z_data.std()
    #     print("===Starting Training===\n")
    #     self.autoencoder.fit(Z_data.values, Y_data.values, batch_size=self.batch_size, epochs=self.epochs, validation_split=0)
    #
    #     test_Y = []
    #
    #     load_time = time.time()
    #     for _ in range(200000, 202599):
    #         if _ % 500 == 0:
    #             print("{} / 100000".format(_), end='\r')
    #         img = cv2.imread(imgs[_])
    #         img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    #         test_Y.append(img.astype("float32") / 255.0)
    #
    #     print("[+] done loading test - took {:.2f} seconds".format(load_time - time.time()))
    #
    #     test_Y = np.array(test_Y)
    #     mean = test_Y.mean()
    #     std = test_Y.std()
    #     test_Z = (test_Y - mean) / std
    #
    #     pred = self.autoencoder.predict(test_Z)
    #     temp = r(0, 2599)
    #     print(temp)
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(test_Y[temp])
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(test_Z[temp])
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(pred[temp])
    #     cv2.imshow("generated", pred[temp])
    #
    #     # if os.path.exists()
    #     self.autoencoder.save_weights(self.model_name + ".h5")

    def generate(self, image=None):
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if image is None:
            img = np.random.normal(size=(9, self.input_size, self.input_size, 3))

            prediction = self.autoencoder.predict(img)

            op = np.vstack((np.hstack((prediction[0], prediction[1], prediction[2])),
                            np.hstack((prediction[3], prediction[4], prediction[5])),
                            np.hstack((prediction[6], prediction[7], prediction[8]))))
            print(op.shape)
            op = cv2.resize(op, (self.input_size * 9, self.input_size * 9), interpolation=cv2.INTER_AREA)
            op = cv2.cvtColor(op, cv2.COLOR_BGR2RGB)
            cv2.imshow("generated", op)
            cv2.imwrite(self.sample_dir + "generated" + str(r(0, 9999)) + ".jpg", (op * 255).astype("uint8"))

        else:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            img = img.astype("float32")
            img = img / 255

            prediction = self.autoencoder.predict(img.reshape(1, self.input_size, self.input_size, 3))
            img = cv2.resize(prediction[0][:, :, ::-1], (960, 960), interpolation=cv2.INTER_AREA)

            # op = cv2.cvtColor(op, cv2.COLOR_BGR2RGB)
            # img = (img - self.mean) / self.std
            # cv2.imshow("prediction", cv2.resize(img/255, (960, 960), interpolation=cv2.INTER_AREA))

            cv2.imshow("prediction", img)

            cv2.imwrite(self.sample_dir + "generated" + str(r(0, 9999)) + ".jpg", (img * 255).astype("uint8"))

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
    elif choice == 'g+':
        # vae.generate(image='img.jpg')
        vae.generate(image='img2.jpg')


if __name__ == "__main__":
    main()

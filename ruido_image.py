#modulos 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras.datasets import mnist
##################################################

class vision_com:
    def __init__(self):
        pass

    def vision_c(self):
        # Cargar el conjunto de datos MNIST
        #minist1 = mnist.load_data()
        #print(minist1)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #visualización de datos
        print(x_train.shape, y_train.shape,'cantidad y tamaño de imagenes')

        i = random.randint(1, 60000)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f'Label: {y_train[i]}')
        #plt.show()

        #Agregando ruido a las imagenes

        #Normalizacion
        x_train = x_train/255
        y_train = y_train/255

        add_noise = np.random.randn(*(28,28))
        noise_factor = 0.3
        added_noise = noise_factor * add_noise
        plt.imshow(added_noise, cmap='gray')
        #plt.show()

        #############################################################
        noise_factor = 0.2
        sample_image =x_train[101]
        noisy_sample_image = sample_image + noise_factor * add_noise

        plt.imshow(noisy_sample_image, cmap='gray')
        #plt.show()
        print(noisy_sample_image.max())
        print(noisy_sample_image.min())

        noisy_sample_image = np.clip(noisy_sample_image, 0., 1.)

        print(noisy_sample_image.max())
        print(noisy_sample_image.min())

        #plt.imshow(noisy_sample_image, cmap='gray')
        #plt.show()
        plt.imshow(noisy_sample_image, cmap='gray')
        plt.show()
        ########################################3
        ## Mejorando la calidad de las imagenes
        x_train_noisy =[]
        noise_facto = 0.2
        for sample_image in x_train:
            sample_image_noisy = sample_image + noise_facto * add_noise
            sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
            x_train_noisy.append(sample_image_noisy)
        
        #combirtiendo set de datos a numpy array
        x_train_noisy = np.array(x_train_noisy)
        #print(x_train_noisy)
        print(x_train_noisy.shape)
        #print(x_test_noisy.shape()) 
        plt.imshow(x_train_noisy[22], cmap='gray')
        plt.show()
        

        x_test_noisy =[]
        noise_facto = 0.4
        for sample_image in x_train:
            sample_image_noisy = sample_image + noise_facto * add_noise
            sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
            x_test_noisy.append(sample_image_noisy)

        x_test_noisy = np.array(x_test_noisy)
        print(x_test_noisy.shape)

        plt.imshow(x_test_noisy[10], cmap='gray')
        plt.show()

        #modelo
        auto_encoder = tf.keras.Sequential()
        #armando capa convolucional
        auto_encoder.add(tf.keras.layers.Conv2D(16, (3, 3), strides=1,padding= "same",input_shape=(28, 28, 1)))
        auto_encoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

        auto_encoder.add(tf.keras.layers.Conv2D(8, (3, 3), strides=1,padding= "same"))
        auto_encoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

        #imagen codificadq

        auto_encoder.add(tf.keras.layers.Conv2D(8, (3, 3), strides=1,padding= "same"))

        #armando decodificador
        auto_encoder.add(tf.keras.layers.UpSampling2D((2, 2)))
        auto_encoder.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=1,padding= "same"))

        auto_encoder.add(tf.keras.layers.UpSampling2D((2, 2)))
        auto_encoder.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=1,activation='sigmoid',padding= "same"))

        #compilado
        auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

        auto_encoder.summary()

        auto_encoder.fit(x_train_noisy.reshape(-1, 28, 28, 1),
                          x_train.reshape(-1, 28, 28, 1), 
                          epochs=10, batch_size=200)
        
        denoised_images = auto_encoder.predict(x_test_noisy.reshape(-1, 28, 28, 1))
        print(denoised_images.shape)

        fig, axes = plt.subplots(nrows=2, ncols=15, figsize=(30, 6))
        for images,row in zip([x_test_noisy[:15], denoised_images], axes):
            for img, ax in zip(images, row):
                ax.imshow(img.reshape(28, 28), cmap='gray')
        
        plt.show()
                


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, concatenate

from utils import *



# denoising autoencoder class that initializes from a list number of layers and nodes

class DenoisingAutoEncoder(Model):
    def __init__(self, latent_dim, shape, n_layers, nodes):
        super(DenoisingAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        if self.latent_dim == None:
            self.latent_dim = 1000
        self.shape = shape
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

        for i in range(n_layers):
            if i == 0:
                self.encoder.add(layers.Dense(nodes[i], activation="elu"))
                self.decoder.add(layers.Dense(nodes[i], activation="elu"))
            elif i == n_layers - 1:
                self.encoder.add(layers.Dense(self.latent_dim, activation="elu"))
                self.decoder.add(layers.Dense(tf.math.reduce_prod(shape), activation="linear"))
            else:
                self.encoder.add(layers.Dense(nodes[i], activation="elu"))
                self.decoder.add(layers.Dense(nodes[i], activation="elu"))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class DenoisingConvAutoEncoder(Model):
    def __init__(self, shape):
        super(DenoisingConvAutoEncoder, self).__init__()
        self.shape = shape
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

        self.encoder.add(Conv1D(256, 3, activation='relu', padding='same', input_shape=(self.shape, 1)))
        self.encoder.add(Conv1D(128, 3, activation='relu', padding='same'))

        
        self.decoder.add(Conv1DTranspose(128, 3, activation='relu', padding='same'))
        self.decoder.add(Conv1DTranspose(256, 3, activation='relu', padding='same'))




    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class DenoisingUNET(Model):
    def __init__(self, num_filters=64, output_channels=1):
        super(DenoisingUNET, self).__init__()
        self.num_filters = num_filters
        self.output_channels = output_channels

        # Encoder (Downsampling)
        self.conv1 = layers.Conv1D(num_filters, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(num_filters * 2, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2)
        self.conv3 = layers.Conv1D(num_filters * 4, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling1D(2)

        # Bottleneck
        self.conv4 = layers.Conv1D(num_filters * 8, 3, activation='relu', padding='same')

        # Decoder (Upsampling)
        self.upconv1 = layers.Conv1DTranspose(num_filters * 4, 2, strides=2, activation='relu', padding='same')
        self.conv5 = layers.Conv1D(num_filters * 4, 3, activation='relu', padding='same')
        self.upconv2 = layers.Conv1DTranspose(num_filters * 2, 2, strides=2, activation='relu', padding='same')
        self.conv6 = layers.Conv1D(num_filters * 2, 3, activation='relu', padding='same')
        self.upconv3 = layers.Conv1DTranspose(num_filters, 2, strides=2, activation='relu', padding='same')
        self.conv7 = layers.Conv1D(num_filters, 3, activation='relu', padding='same')

        # Output layer
        self.output_conv = layers.Conv1D(output_channels, 1, activation='linear')

    def call(self, inputs):
        # Encoder
        c1 = self.conv1(inputs)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        # Bottleneck
        b = self.conv4(p3)

        # Decoder
        u1 = self.upconv1(b)
        concat1 = layers.concatenate([u1, c3])
        c5 = self.conv5(concat1)
        u2 = self.upconv2(c5)
        concat2 = layers.concatenate([u2, c2])
        c6 = self.conv6(concat2)
        u3 = self.upconv3(c6)
        concat3 = layers.concatenate([u3, c1])
        c7 = self.conv7(concat3)

        # Output
        outputs = self.output_conv(c7)
        return outputs

    

def main(continue_training = False,denoising = True, conv_layer = True, ensemble = False, batch_size = 256,train_years = [2015,2016,2017,2018,2019], test_year = 2020, seed_offset = 5,save_path='/home/wkeely/OCO_L2DIA/'):
    # load the training data
    load_path = '/scratch-science/algorithm/wkeely/'
    training_band_obs = []
    training_band_no_eof = []
    training_band_sim = []
    train_states = []
    test_band_obs = []
    test_band_no_eof = []
    test_band_sim = []
    test_states = []
    for year in train_years:
        file = pickle.load(open(load_path + 'L2DiaND_XCO2_'+str(year)+'_downsampled_eof_removed_aligned.p', 'rb'))
        training_band_obs.append(file['o2_band_obs'])
        training_band_no_eof.append(file['o2_band_no_eof'])
        training_band_sim.append(file['o2_band'])
        train_states.append(file['states'])
        state_names = file['state_var']

    # load the test data
    file = pickle.load(open(load_path + 'L2DiaND_XCO2_'+str(test_year)+'_downsampled_eof_removed_aligned.p', 'rb'))
    test_band_obs.append(file['o2_band_obs'])
    test_band_no_eof.append(file['o2_band_no_eof'])
    test_band_sim.append(file['o2_band'])
    test_states.append(file['states'])

    # vstack the data
    training_band_obs = np.vstack(training_band_obs)
    training_band_no_eof = np.vstack(training_band_no_eof)
    training_band_sim = np.vstack(training_band_sim)
    train_states = np.vstack(train_states)
    test_band_obs = np.vstack(test_band_obs)
    test_band_no_eof = np.vstack(test_band_no_eof)
    test_band_sim = np.vstack(test_band_sim)
    test_states = np.vstack(test_states)

    print('training_band_obs shape: ', training_band_obs.shape)
    print('training_band_no_eof shape: ', training_band_no_eof.shape)
    print('training_band_sim shape: ', training_band_sim.shape)
    print('test_band_obs shape: ', test_band_obs.shape)
    print('test_band_no_eof shape: ', test_band_no_eof.shape)
    print('test_band_sim shape: ', test_band_sim.shape)

    # select rows with outcome_flag == 1
    idx = state_names.index('RetrievalResults/outcome_flag')
    outcome_flag = train_states[:, idx]

    training_band_obs = training_band_obs[outcome_flag == 1, :]
    training_band_no_eof = training_band_no_eof[outcome_flag == 1, :]
    training_band_sim = training_band_sim[outcome_flag == 1, :]

    outcome_flag = test_states[:, idx]

    test_band_obs = test_band_obs[outcome_flag == 1, :]
    test_band_no_eof = test_band_no_eof[outcome_flag == 1, :]
    test_band_sim = test_band_sim[outcome_flag == 1, :]

    # remove nan rows
    nan_rows = np.argwhere(np.isnan(training_band_obs))
    training_band_obs = np.delete(training_band_obs, nan_rows, axis=0)
    training_band_no_eof = np.delete(training_band_no_eof, nan_rows, axis=0)
    training_band_sim = np.delete(training_band_sim, nan_rows, axis=0)
    nan_rows = np.argwhere(np.isnan(test_band_obs))
    test_band_obs = np.delete(test_band_obs, nan_rows, axis=0)
    test_band_no_eof = np.delete(test_band_no_eof, nan_rows, axis=0)
    test_band_sim = np.delete(test_band_sim, nan_rows, axis=0)

    # scale by 1E-19
    training_band_obs = training_band_obs * 1E-19
    training_band_no_eof = training_band_no_eof * 1E-19
    training_band_sim = training_band_sim * 1E-19
    test_band_obs = test_band_obs * 1E-19
    test_band_no_eof = test_band_no_eof * 1E-19
    test_band_sim = test_band_sim * 1E-19

    

    if conv_layer:
        training_band_obs = np.reshape(training_band_obs, (training_band_obs.shape[0], training_band_obs.shape[1], 1))
        training_band_no_eof = np.reshape(training_band_no_eof, (training_band_no_eof.shape[0], training_band_no_eof.shape[1], 1))
        training_band_sim = np.reshape(training_band_sim, (training_band_sim.shape[0], training_band_sim.shape[1], 1))
        test_band_obs = np.reshape(test_band_obs, (test_band_obs.shape[0], test_band_obs.shape[1], 1))
        test_band_no_eof = np.reshape(test_band_no_eof, (test_band_no_eof.shape[0], test_band_no_eof.shape[1], 1))
        test_band_sim = np.reshape(test_band_sim, (test_band_sim.shape[0], test_band_sim.shape[1], 1))

        # model = DenoisingConvAutoEncoder(shape=training_band_no_eof.shape[1])
        model = DenoisingUNET()

    else:
        model = DenoisingAutoEncoder(latent_dim=128, shape=shape, n_layers=6, nodes=[512,258,258,128,128])


    

    # standardize the data
    my_obs = np.mean(training_band_obs, axis=0)
    sy_obs = np.std(training_band_obs, axis=0)
    maxy_obs = np.max(np.abs((training_band_obs - my_obs)/sy_obs))



    my_no_eof = np.mean(training_band_no_eof, axis=0)
    sy_no_eof = np.std(training_band_no_eof, axis=0)
    maxy_no_eof = np.max(np.abs((training_band_no_eof - my_no_eof)/sy_no_eof))

    my_sim = np.mean(training_band_sim, axis=0)
    sy_sim = np.std(training_band_sim, axis=0)
    maxy_sim = np.max(np.abs((training_band_sim - my_sim)/sy_sim))




    training_band_obs = (training_band_obs - my_obs)/(sy_obs * maxy_obs)
    training_band_no_eof = (training_band_no_eof - my_no_eof)/(sy_no_eof * maxy_no_eof)
    test_band_obs = (test_band_obs - my_obs)/(sy_obs * maxy_obs)
    test_band_no_eof = (test_band_no_eof - my_no_eof)/(sy_no_eof * maxy_no_eof)
    test_band_sim = (test_band_sim - my_sim)/(sy_sim * maxy_sim)

    if denoising:
        Xtrain = training_band_obs
        ytrain = training_band_no_eof
        Xtest = test_band_obs
        ytest = test_band_no_eof
        if conv_layer:
            model_str = '_denoising_inference_conv_'
        else:
            model_str = '_denoising_inference_vanilla_'
    else:
        Xtrain = training_band_no_eof
        ytrain = training_band_obs
        Xtest = test_band_no_eof
        ytest = test_band_obs
        if conv_layer:
            model_str = '_denoising_inference_conv_'
        else:
            model_str = '_denoising_inference_vanilla_'




     # callback that tracks rmse of val each epoch
    class RMSECallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('RMSE for val: ', np.sqrt(logs['val_loss']))
    # class ChiSquaredCallback(tf.keras.callbacks.Callback): TODO: implement chi squared loss
    #     def on_epoch_end(self, epoch, logs=None):
    #         print('Chi Squared for val: ', np.sum((ytest - self.model.predict(Xtest))**2))
    if ensemble:    
        print('training ensemble')
        rmse_EOF = np.sqrt(np.mean((test_band_obs - test_band_sim)**2))
        rmse_no_EOF = np.sqrt(np.mean((test_band_no_eof - test_band_obs)**2))
        print('RMSE for EOF: ', rmse_EOF)
        print('RMSE for no EOF: ', rmse_no_EOF)
        for i in range(3):
            # set randome seed for initializing weights
            seed_o2 = i+seed_offset
            opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
            model = DenoisingUNET()
            model.compile(optimizer=opt, loss=losses.MeanSquaredError())
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + 'DAE'+model_str+'o2_'+str(seed_o2)+'.h5', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
            model.fit(Xtrain,ytrain, epochs=40, batch_size=batch_size, shuffle=True, validation_data=(Xtest, ytest), callbacks=[RMSECallback(),  model_checkpoint_callback])
            test_pred = model.predict(Xtest)
            rmse_DAE = np.sqrt(np.mean((test_band_no_eof - test_pred)**2))
            
            print('RMSE for DAE: ', rmse_DAE)
            print('RMSE for EOF: ', rmse_EOF)
    else:
        print('training single model')
        rmse_EOF = np.sqrt(np.mean((test_band_obs - test_band_sim)**2))
        rmse_no_EOF = np.sqrt(np.mean((test_band_no_eof - test_band_obs)**2))
        print('RMSE for EOF: ', rmse_EOF)
        print('RMSE for no EOF: ', rmse_no_EOF)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model = DenoisingUNET()
        # initalize the model
        model.compile(optimizer=opt, loss=losses.MeanSquaredError())

        # model checkpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + 'DAE'+model_str+'o2.h5', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + 'DAE'+model_str+'o2_'+str(seed_o2)+'.h5', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        model.fit(Xtrain,ytrain, epochs=40, batch_size=batch_size, shuffle=True, validation_data=(Xtest, ytest), callbacks=[RMSECallback(),  model_checkpoint_callback])
        test_pred = model.predict(Xtest)
        rmse_DAE = np.sqrt(np.mean((test_band_no_eof - test_pred)**2))
        
        print('RMSE for DAE: ', rmse_DAE)
        print('RMSE for EOF: ', rmse_EOF)        



    
    if continue_training:
        print('loading weights')
        model.build((None, shape, 1))
        model.load_weights(save_path + 'DAE'+model_str+'wco2.h5')

        # train the model
        model.fit(Xtrain,ytrain, epochs=100, batch_size=batch_size, shuffle=True, validation_data=(Xtest, ytest), callbacks=[RMSECallback(),  model_checkpoint_callback])
        # print rmse of the test data
    
    print('done')

if __name__ == "__main__":
    main()



    







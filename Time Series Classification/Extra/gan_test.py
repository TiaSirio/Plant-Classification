from google.colab import drive
drive.mount('/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /gdrive/My Drive/ChallengeANN/Challenge2

"""### Install missing libraries"""

!pip install visualkeras
import visualkeras

"""### Import libraries"""

import tensorflow as tf
import numpy as np
import os
import shutil
import random
import glob
import imageio
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

"""### Set seed for reproducibility"""

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

"""### Suppress warnings"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""### Load and process data"""

import os.path
from os import path

datasetX = "x_train.npy"
datasetY = "y_train.npy"

# Load the dataset to be used for classification
if (not os.path.exists(datasetX)) or (not os.path.exists(datasetY)):
    !unzip training_dataset_homework2.zip

label_mapping = {
    'Wish': 0,
    'Another': 1,
    'Comfortably': 2,
    'Money': 3,
    'Breathe': 4,
    'Time': 5,
    'Brain': 6,
    'Echoes': 7,
    'Wearing': 8,
    'Sorrow': 9,
    'Hey': 10,
    'Shine': 11
}

datasetX = np.load('x_train.npy')
datasetY = np.load('y_train.npy')
#describe_array(datasetX)
#describe_array(datasetY)
print(datasetX.shape)
print(datasetY.shape)

#Convert a 36x6 window in a 36x36 window that can be seen as a squared image
def windowToImg(app):
  ret = np.zeros((36,36),dtype=app.dtype)
  for rowIndex,row in enumerate(app):
    for columnIndex,column in enumerate(row):
      for i in range(6):
        ret[rowIndex][columnIndex+i*6] = app[rowIndex][columnIndex] # Just copying 6 times the 6 features columns
  return ret.T

from scipy.sparse import data
# Oversampling

classesA = [None for i in range(12)]
for i in np.ndindex(datasetX.shape[:1]):

    for j in range(12):
        if datasetY[i] == j:
          if classesA[j] is None:
              classesA[j] = datasetX[i][None]
          else:
              classesA[j] = np.vstack((classesA[j],datasetX[i][None]))

firstVal = True

newDataSetX = np.zeros((datasetX.shape[0],36,36),dtype=datasetX.dtype)
newDataSetY = np.zeros(datasetY.shape[0],dtype=datasetX.dtype)

count = 0
for idx,i in enumerate(classesA):
  lu = len(i) if i is not None else 0
  for app in i:
    app = windowToImg(app)
    newDataSetX[count] = app
    newDataSetY[count] = idx
    count = count + 1

datasetX = newDataSetX
datasety = newDataSetY

datasetX = np.expand_dims(((datasetX-127.5)/127.5),axis=-1).astype('float32')
X = datasetX
y = datasetY

!pip install pyts

from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt

Xtrain = X # both N x T time series samples

mtf = MarkovTransitionField(image_size=24)
im_train = mtf.fit_transform(X)
#im_test = mtf.transform(Xtest)

# plot one image
plt.imshow(im_train[0])
plt.show()

# Inspect the data
fig, axes = plt.subplots(1, 10, figsize=(20,10))
for i in range(10):
    ax = axes[i%10]
    ax.imshow(np.squeeze(X[i].T), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

print('Data shape', X.shape)
print('Data min {:0.2f}\nData max {:0.2f}\nData mean {:0.2f}\nData std {:0.2f}'.format(
    X.min(), X.max(), X.mean(), X.std()))

input_shape = X.shape[1:]
input_shape

"""### Vanilla Generative Adversarial Network"""

def get_dc_discriminator(input_shape, seed=seed):
    tf.random.set_seed(seed)

    # Build the discriminator layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    x = tfkl.ZeroPadding2D((2,2), name='padding')(input_layer)

    x = tfkl.Conv2D(64, 3, padding='same', strides=2, name='conv1')(x)
    x = tfkl.BatchNormalization(name='bn1')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation1')(x)

    x = tfkl.Conv2D(128, 3, padding='same', strides=2, name='conv2')(x)
    x = tfkl.BatchNormalization(name='bn2')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation2')(x)

    x = tfkl.Conv2D(256, 3, padding='same', strides=2, name='conv3')(x)
    x = tfkl.BatchNormalization(name='bn3')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation3')(x)

    x = tfkl.GlobalAveragePooling2D(name='gap')(x)
    x = tfkl.Dropout(.5, seed=seed, name='dropout')(x)
    x = tfkl.Dense(1, name='dense_out')(x)
    output_layer = tfkl.Activation('sigmoid', name='output_layer')(x)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='discriminator')

    # Return the discriminator
    return model
discriminator = get_dc_discriminator(input_shape)
discriminator.summary()
display(visualkeras.layered_view(discriminator, legend=True, scale_xy=6))
tfk.utils.plot_model(discriminator, show_shapes=True, expand_nested=True, to_file='vanilla_discriminator.png')

latent_dim = 128

def get_dc_generator(input_shape, seed=seed):
    tf.random.set_seed(seed)

    # Build the generator layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    x = tfkl.Dense(5*5*64, use_bias=False, name='dense0')(input_layer)
    x = tfkl.BatchNormalization(name='bn0')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation0')(x)
    x = tfkl.Reshape((5,5,64))(x)

    x = tfkl.UpSampling2D(name='upsampling1')(x)
    x = tfkl.Conv2D(64, 3, padding='same', use_bias=False, name='conv1')(x)
    x = tfkl.BatchNormalization(name='bn1')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation1')(x)

    x = tfkl.UpSampling2D(name='upsampling2')(x)
    x = tfkl.Conv2D(128, 3, padding='same', use_bias=False, name='conv2')(x)
    x = tfkl.BatchNormalization(name='bn2')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation2')(x)

    x = tfkl.UpSampling2D(name='upsampling3')(x)
    x = tfkl.Conv2D(256, 3, padding='same', use_bias=False, name='conv3')(x)
    x = tfkl.BatchNormalization(name='bn3')(x)
    x = tfkl.LeakyReLU(alpha=0.2, name='activation3')(x)

    x = tfkl.Conv2D(1, 3, padding='same', use_bias=False, name='conv_out')(x)
    x = tfkl.Activation('tanh', name='activation_out')(x)
    output_layer = tfkl.Cropping2D((2,2), name='cropping')(x)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='generator')

    # Return the model
    return model
generator = get_dc_generator(latent_dim)
generator.summary()
display(visualkeras.layered_view(generator, legend=True, scale_xy=6))
tfk.utils.plot_model(generator, show_shapes=True, expand_nested=True, to_file='vanilla_generator.png')

class GAN(tfk.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.d_loss_tracker = tfk.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tfk.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.d_loss_tracker,
            self.g_loss_tracker
        ]

    @tf.function
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images from z
        generated_images = self.generator(z)

        # Combine generated images and real ones
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Create labels so that fake images correspond to class 0 and real images to class 1
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)


        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = tf.reduce_mean(tfk.losses.binary_crossentropy(labels, predictions))
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        loss = d_loss

        # Sample random points in the latent space
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Create misleading labels for fake images so that they correspond to class 1
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            misleading_predictions = self.discriminator(self.generator(z))
            g_loss = tf.reduce_mean(tfk.losses.binary_crossentropy(misleading_labels, misleading_predictions))
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        loss += g_loss

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "loss": self.loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

class GANMonitor(tfk.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=latent_dim, name='', gray=False):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.name = name
        self.gray = gray

    def on_epoch_end(self, epoch, logs=None):
        tf.random.set_seed(seed)
        os.makedirs(self.name+'temp', exist_ok=True)
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors).numpy()

        fig, axes = plt.subplots(1, self.num_img, figsize=(20,self.num_img))
        for i in range(self.num_img):
            img = tfk.preprocessing.image.array_to_img(generated_images[i])
            ax = axes[i%self.num_img]
            if self.gray:
                ax.imshow(np.squeeze(img), cmap='gray')
            else:
                ax.imshow(np.squeeze(img))
            ax.axis('off')
        fig.savefig(self.name+'temp/'+'{:0>5}'.format(epoch)+'.png')
        plt.tight_layout()
        plt.show()

    def on_train_end(self, logs=None):
        fp_in = self.name+"temp/*.png"
        fp_out = self.name+"_generated_images.gif"
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=100, optimize=False)
        shutil.rmtree(self.name+"temp")

epochs = 100
learning_rate = 5e-5
batch_size = 128

gan = GAN(
    discriminator = get_dc_discriminator(input_shape),
    generator = get_dc_generator(latent_dim),
    latent_dim = latent_dim
)
gan.compile(
    d_optimizer = tfk.optimizers.Adam(learning_rate=learning_rate),
    g_optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
)

def sample(model, num_img, latent_dim, fixed=True, gray=False):
    if fixed:
        tf.random.set_seed(seed)
    z = tf.random.normal(shape=(num_img, latent_dim))
    generated_images = model(z).numpy()

    fig, axes = plt.subplots(1, num_img, figsize=(20,num_img))
    for i in range(num_img):
        img = tfk.preprocessing.image.array_to_img(generated_images[i])
        ax = axes[i%num_img]
        if gray:
            ax.imshow(np.squeeze(img), cmap='gray')
        else:
            ax.imshow(np.squeeze(img))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_dataset(model, num_img, latent_dim, fixed=True):
    if fixed:
        tf.random.set_seed(seed)
    z = tf.random.normal(shape=(num_img, latent_dim))
    generated_images = model(z).numpy()
    return generated_images

sample(gan.generator, 10, latent_dim, fixed=False, gray=True)



history = gan.fit(
    X,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = [GANMonitor(name='vanilla', gray=True)],
    verbose = 2
).history

# Plot the training
plt.figure(figsize=(20,5))
plt.plot(history['loss'], label='GAN loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)

plt.figure(figsize=(20,5))
plt.plot(history['d_loss'], label='Discriminator loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)

plt.figure(figsize=(20,5))
plt.plot(history['g_loss'], label='Generator loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)
plt.show()

gan.generator.save('vanilla_gan_generator')

vanilla_gan_generator = tfk.models.load_model('vanilla_gan_generator')

sample(vanilla_gan_generator, 10, latent_dim, fixed=False, gray=True)
sample(vanilla_gan_generator, 10, latent_dim, fixed=False, gray=True)

imgs = 2500
X_hat = generate_dataset(vanilla_gan_generator, imgs, latent_dim)*127.5 + 127.5
X_hat = np.reshape(X_hat, (X_hat.shape[0],X_hat.shape[1]*X_hat.shape[2]))
X_hat = np.mean(X_hat, axis=0)

X_temp = X[:imgs]*127.5 + 127.5
X_temp = np.reshape(X_temp, (X_temp.shape[0],X_temp.shape[1]*X_temp.shape[2]))
X_temp = np.mean(X_temp, axis=0)

print(X_hat.shape, X_temp.shape)

plt.figure(figsize=(10,4))
sns.kdeplot(x=X_hat, label='Sampled data', linewidth=3)
sns.kdeplot(x=X_temp, label='Real data', linewidth=3)
plt.legend()
plt.show()

"""### Deep Convolutional Conditional GAN"""

y = datasetY

image_size = input_shape[0]
num_classes = len(np.unique(y))
num_channels = input_shape[-1]

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

conditional_generator_input = (generator_in_channels)
conditional_discriminator_input = (image_size, image_size, discriminator_in_channels)
conditional_discriminator_input, conditional_generator_input

discriminator = get_dc_discriminator(conditional_discriminator_input)
discriminator.summary()
display(visualkeras.layered_view(discriminator, legend=True, scale_xy=6))
tfk.utils.plot_model(discriminator, show_shapes=True, expand_nested=True, to_file='conditional_discriminator.png')

generator = get_dc_generator(conditional_generator_input)
generator.summary()
display(visualkeras.layered_view(generator, legend=True, scale_xy=6))
tfk.utils.plot_model(generator, show_shapes=True, expand_nested=True, to_file='conditional_generator.png')

class ConditionalGAN(tfk.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.d_loss_tracker = tfk.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tfk.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.d_loss_tracker,
            self.g_loss_tracker
        ]

    @tf.function
    def train_step(self, data):
        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=[image_size * image_size])
        image_one_hot_labels = tf.reshape(image_one_hot_labels, (-1, image_size, image_size, num_classes))

        # Sample random points in the latent space
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        z = tf.concat([z, one_hot_labels], -1)

        # Generate fake images from z
        generated_images = self.generator(z)


        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        # Combine generated images and real ones
        combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)

        # Create labels so that fake images correspond to class 0 and real images to class 1
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = tf.reduce_mean(tfk.losses.binary_crossentropy(labels, predictions))
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        loss = d_loss

        # Sample random points in the latent space
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        z = tf.concat([z, one_hot_labels], axis=1)

        # Create misleading labels for fake images so that they correspond to class 1
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(z)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            misleading_predictions = self.discriminator(fake_image_and_labels)
            g_loss = tf.reduce_mean(tfk.losses.binary_crossentropy(misleading_labels, misleading_predictions))
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        loss += g_loss

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "loss": self.loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

class ConditionalGANMonitor(tfk.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=latent_dim, name='', gray=False):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.name = name
        self.gray = gray

    def on_epoch_end(self, epoch, logs=None):
        tf.random.set_seed(seed)
        os.makedirs(self.name+'temp', exist_ok=True)
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        labels = tf.cast(tf.math.floormod(tf.range(0,self.num_img), num_classes), 'float32')
        labels = tfk.utils.to_categorical(labels, num_classes)
        random_latent_vectors = tf.concat([random_latent_vectors,labels],-1)
        generated_images = self.model.generator(random_latent_vectors).numpy()

        fig, axes = plt.subplots(1, self.num_img, figsize=(20,2*self.num_img))
        for i in range(self.num_img):
            img = tfk.preprocessing.image.array_to_img(generated_images[i])
            ax = axes[i%self.num_img]
            if self.gray:
                ax.imshow(np.squeeze(img), cmap='gray')
            else:
                ax.imshow(np.squeeze(img))
            ax.axis('off')
        fig.savefig(self.name+'temp/'+'{:0>5}'.format(epoch)+'.png')
        plt.tight_layout()
        plt.show()

    def on_train_end(self, logs=None):
        fp_in = self.name+"temp/*.png"
        fp_out = self.name+"_generated_images.gif"
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=100, optimize=False)
        shutil.rmtree(self.name+"temp")

all_labels = tfk.utils.to_categorical(y, num_classes)
dataset = tf.data.Dataset.from_tensor_slices((X, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

cgan = ConditionalGAN(
    discriminator = get_dc_discriminator(conditional_discriminator_input),
    generator = get_dc_generator(conditional_generator_input),
    latent_dim = latent_dim
)
cgan.compile(
    d_optimizer = tfk.optimizers.Adam(learning_rate=learning_rate),
    g_optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
)

def conditional_sample(model, num_img, latent_dim, fixed=True, gray=False, label=None):
    if fixed:
        tf.random.set_seed(seed)
    z = tf.random.normal(shape=(num_img, latent_dim))
    if label == None:
        labels = tf.cast(tf.math.floormod(tf.range(0,num_img), num_classes), 'float32')
    else:
        labels = tf.cast(tf.math.floormod(tf.ones(num_img)*label, num_classes), 'float32')
    labels = tfk.utils.to_categorical(labels, num_classes)
    z = tf.concat([z,labels],-1)
    generated_images = model(z).numpy()

    fig, axes = plt.subplots(1, num_img, figsize=(20,2*num_img))
    for i in range(num_img):
        img = tfk.preprocessing.image.array_to_img(generated_images[i])
        ax = axes[i%num_img]
        if gray:
            ax.imshow(np.squeeze(img), cmap='gray')
        else:
            ax.imshow(np.squeeze(img))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_conditional_dataset(model, num_img, latent_dim, fixed=True, label=None):
    if fixed:
        tf.random.set_seed(seed)
    z = tf.random.normal(shape=(num_img, latent_dim))
    if label == None:
        labels = tf.cast(tf.math.floormod(tf.range(0,num_img), num_classes), 'float32')
    else:
        labels = tf.cast(tf.math.floormod(tf.ones(num_img)*label, num_classes), 'float32')
    labels = tfk.utils.to_categorical(labels, num_classes)
    z = tf.concat([z,labels],-1)
    generated_images = model(z).numpy()
    return generated_images, labels

conditional_sample(cgan.generator, 10, latent_dim, gray=True)

c_history = cgan.fit(
    dataset,
    epochs = epochs,
    callbacks = [ConditionalGANMonitor(name='conditional', gray=True)],
    verbose = 2
).history

# Plot the training
plt.figure(figsize=(20,5))
plt.plot(c_history['loss'], label='GAN loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)

plt.figure(figsize=(20,5))
plt.plot(c_history['d_loss'], label='Discriminator loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)

plt.figure(figsize=(20,5))
plt.plot(c_history['g_loss'], label='Generator loss', alpha=.8, linewidth=3)
plt.legend(loc='upper left')
plt.grid(alpha=.3)
plt.show()

cgan.generator.save('conditional_gan_generator')

conditional_gan_generator = tfk.models.load_model('conditional_gan_generator')

conditional_sample(conditional_gan_generator, 10, latent_dim, fixed=False, gray=True)
conditional_sample(conditional_gan_generator, 10, latent_dim, fixed=False, gray=True)

imgs = 2500
X_hat, y_hat = generate_conditional_dataset(conditional_gan_generator, imgs, latent_dim)
X_hat = np.reshape(X_hat, (X_hat.shape[0],X_hat.shape[1]*X_hat.shape[2]))*127.5 + 127.5
X_hat = np.mean(X_hat, axis=0)
y_hat = np.argmax(y_hat,axis=1)

X_temp = X[:imgs]*127.5 + 127.5
X_temp = np.reshape(X_temp, (X_temp.shape[0],X_temp.shape[1]*X_temp.shape[2]))
X_temp = np.mean(X_temp, axis=0)
y_temp = y[:imgs]

print(X_hat.shape, X_temp.shape)

plt.figure(figsize=(10,4))
sns.kdeplot(x=X_hat, label='Sampled data', linewidth=3)
sns.kdeplot(x=X_temp, label='Real data', linewidth=3)
plt.legend()
plt.show()

imgs = 1000
fig, axes = plt.subplots(2, 5, figsize=(28,12), sharey=True, sharex=True)
fig.suptitle('Classes pixels distributions')
for i in range(10):
    X_hat, y_hat = generate_conditional_dataset(conditional_gan_generator, imgs, latent_dim, label=i)
    X_hat = np.reshape(X_hat, (X_hat.shape[0],X_hat.shape[1]*X_hat.shape[2]))*127.5 + 127+5
    X_hat = np.mean(X_hat, axis=0)
    X_temp = X[y==i][:imgs]*127.5 + 127+5
    X_temp = np.reshape(X_temp, (X_temp.shape[0],X_temp.shape[1]*X_temp.shape[2]))
    X_temp = np.mean(X_temp, axis=0)
    axes[i//5, i%5].set_title(f'Class {i}')
    sns.kdeplot(ax=axes[i//5, i%5], x=X_hat, label='Sampled data', linewidth=2)
    sns.kdeplot(ax=axes[i//5, i%5], x=X_temp, label='Real data', linewidth=2)
plt.legend()
plt.show()

for i in range(num_classes):
    conditional_sample(conditional_gan_generator, 10, latent_dim, gray=True, fixed=False, label=i)

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 500

# Sample noise for the interpolation.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = tfk.utils.to_categorical([first_number], num_classes)
    second_label = tfk.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (first_label * (1 - percent_second_label) + second_label * percent_second_label)

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = conditional_gan_generator.predict(noise_and_labels, verbose=0)
    return fake


start_class = 0
end_class = 1

fake_images = interpolate_class(start_class, end_class)

fake_images = interpolate_class(start_class, end_class)*127.5 + 127.5
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("animation01.gif", converted_images, fps=60)

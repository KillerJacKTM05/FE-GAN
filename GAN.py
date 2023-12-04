import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Embedding, Flatten, Dense, Reshape, Multiply, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Dropout

data_dir = "./RAF-DB"
save_dir = './saved_models/'
img_size = (100, 100)
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_to_label = {emotion: i for i, emotion in enumerate(emotions)}

# Data Generator for efficient batch loading
class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size, img_size, emotions, shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.emotions = emotions
        self.shuffle = shuffle
        self.image_paths, self.labels = self._load_image_paths_labels(data_dir)
        self.on_epoch_end()

    def _load_image_paths_labels(self, data_dir):
        image_paths = []
        labels = []
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(data_dir, f"{i}_{emotion}")
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                image_paths.append(img_path)
                labels.append(i)
        return np.array(image_paths), np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = self.image_paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        images = np.array([np.array(Image.open(img_path).resize(self.img_size)) for img_path in batch_image_paths])
        images = images.astype('float32') / 255.0  # Normalize images

        return images, batch_labels

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

def residual_block(x, filters, kernel_size=3):
    # Save the input value for the shortcut connection
    shortcut = x

    # First convolution
    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    # Second convolution
    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    # Check if the shortcut needs to be processed to match the shape
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add the shortcut to the output of the block (element-wise add)
    y = layers.add([shortcut, y])

    return y


def build_generator(z_dim, img_shape, num_classes=7):
    # Inputs
    input_noise = layers.Input(shape=(z_dim,))
    input_label = layers.Input(shape=(1,), dtype='int32')
    input_real_image = layers.Input(shape=img_shape)

    # Embedding for the label
    label_embedding = layers.Embedding(num_classes, z_dim, input_length=1)(input_label)
    label_embedding = layers.Flatten()(label_embedding)

    # Combine noise and label embedding
    combined_input = layers.multiply([input_noise, label_embedding])

    # Dense layer for combined input
    x = layers.Dense(128 * 25 * 25)(combined_input)
    x = layers.Reshape((25, 25, 128))(x)

    # Apply residual blocks
    x = residual_block(x, 128)
    x = layers.UpSampling2D()(x)
    x = residual_block(x, 64)
    x = layers.UpSampling2D()(x)
    x = residual_block(x, 32)

    # Extract style features from the input image
    style_extractor = build_style_extractor(img_shape)
    style_features = style_extractor(input_real_image)
    # Upsample the style features to match the spatial dimensions of 'x'
    style_features = UpSampling2D(size=(4, 4))(style_features) # correct upsampling from (25, 25) to (100, 100)
    # Combine generator features with style features
    x = layers.Concatenate()([x, style_features])

    # Final convolutions to produce the output image
    x = layers.Conv2D(32, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)

    model = Model([input_noise, input_real_image, input_label], x)
    return model

def build_style_extractor(img_shape):
    # Input layer
    input_img = layers.Input(shape=img_shape)

    # Convolutional layers to extract features
    y = Conv2D(32, kernel_size=3, strides=2, padding='same')(input_img) # shape becomes (50, 50)
    y = LeakyReLU()(y)
    y = Conv2D(64, kernel_size=3, strides=2, padding='same')(y) # shape becomes (25, 25)
    y = LeakyReLU()(y)
    
    # Adjusted to avoid further reduction of spatial dimensions
    y = Conv2D(128, kernel_size=3, padding='same')(y) # shape remains (25, 25)
    y = LeakyReLU()(y)

    # Build and compile the model
    model = Model(input_img, y)
    return model

def build_discriminator(img_shape, num_classes = 7):
    input_img = layers.Input(shape=img_shape)
    input_label = layers.Input(shape=(1,), dtype='int32')
    
    # Embedding for categorical input
    label_embedding = layers.Embedding(num_classes, np.prod(img_shape), input_length=1)(input_label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
    
    # Combine image and label
    x = layers.Concatenate(axis=-1)([input_img, label_embedding])
    
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model([input_img, input_label], x)
    return model

# Create GAN models
img_shape = (100, 100, 3)
z_dim = 100
# Define loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = build_generator(z_dim, img_shape, len(emotions))
discriminator = build_discriminator(img_shape, len(emotions))
# See parameters
generator.summary()
discriminator.summary()


# Get training input
epochs = int(input("Enter the number of epochs: "))
batch_size = int(input("Enter the batch size: "))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Training step function for the discriminator
@tf.function
def train_discriminator(images, labels):
    current_batch_size = len(images)  # Get the current batch size
    noise = np.random.normal(0, 1, (current_batch_size, z_dim))

    # We need to provide real images as input to the generator
    # For simplicity, we can reuse the 'images' from the current batch
    with tf.GradientTape() as disc_tape:
        generated_images = generator([noise, images, labels], training=True)
        
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        
    gradients = disc_tape.gradient(total_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return total_loss

# Training step function for the generator
@tf.function
def train_generator(noise, real_images, labels):
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise, real_images, labels], training=True)
        
        fake_output = discriminator([generated_images, labels], training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        
    gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return gen_loss

# Training loop
d_losses = []
g_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    epoch_d_losses = []
    epoch_g_losses = []
    data_generator = DataGenerator(data_dir, batch_size, img_size, emotions)
    
    for image_batch, label_batch in data_generator:
        noise = np.random.normal(0, 1, (len(image_batch), z_dim))
        
        d_loss = train_discriminator(image_batch, label_batch)
        d_loss = train_discriminator(image_batch, label_batch)
        
        real_images = image_batch
        g_loss = train_generator(noise, real_images, label_batch)
        
        epoch_d_losses.append(d_loss)
        epoch_g_losses.append(g_loss)
        
    # Calculate mean losses for the epoch
    mean_d_loss = np.mean(epoch_d_losses)
    mean_g_loss = np.mean(epoch_g_losses)    
    d_losses.append(mean_d_loss)
    g_losses.append(mean_g_loss) 
    
    # Generate images after the epoch
    noise = tf.random.normal([len(emotions), z_dim])
    labels_tensor = tf.convert_to_tensor(list(range(len(emotions))))

    # Fetch a single random image from the data generator
    data_gen = DataGenerator(data_dir, 1, img_size, emotions, shuffle=True)
    real_image, _ = next(iter(data_gen))  # Fetches one image
    
    # Replicate this image for each emotion label
    real_images = np.array([real_image[0] for _ in range(len(emotions))])

    generated_images = generator([noise, real_images, labels_tensor])

    fig, axs = plt.subplots(1, len(emotions), figsize=(15, 15))
    for i, img in enumerate(generated_images):
        axs[i].imshow((img + 1) / 2)  # Convert from [-1, 1] to [0, 1]
        axs[i].title.set_text(emotions[i])
        axs[i].axis('off')
    plt.show()
    print(f"Discriminator Loss: {mean_d_loss}, Generator Loss: {mean_g_loss}")
    generator.save_weights(os.path.join(save_dir, 'generator_weights.h5'))
    discriminator.save_weights(os.path.join(save_dir, 'discriminator_weights.h5'))
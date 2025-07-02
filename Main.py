import tensorflow as tf
import os
import time
import pydot
from matplotlib import pyplot as plt
from IPython import display

# -------------------------------
# Dataset Preparation and Loading
# -------------------------------

pathMain = os.getcwd()
PATH = pathMain + '\\pictures\\'  # Path to dataset folder

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512

def load(image_file):
    """
    Load an image file and split it into input and target images.
    Assumes input and target are concatenated side-by-side.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

# Example load and display
inp, re = load(PATH + '\\train\\1.jpg')
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)


def resize(input_image, real_image, height, width):
    """Resize input and target images to given size."""
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    """Randomly crop input and target images."""
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    """Normalize images to [-1, 1]."""
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    """
    Apply random jittering for data augmentation:
    - Resize to 512x512
    - Random crop to 512x512 (in your original code, cropping to 512 is redundant but kept for consistency)
    - Random horizontal flip with 50% probability
    """
    input_image, real_image = resize(input_image, real_image, 512, 512)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


# Visualize some augmented samples
plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i+1)
    plt.imshow(rj_inp / 255.0)
    plt.axis('off')


def load_image_train(image_file):
    """Load and preprocess training images with augmentation."""
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    """Load and preprocess test images without augmentation."""
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


# -------------------------
# Dataset Input Pipelines
# -------------------------

train_dataset = tf.data.Dataset.list_files(PATH + 'train\\*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'test\\*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# -------------------------
# Model Building Blocks
# -------------------------

OUTPUT_CHANNELS = 3  # RGB output

def downsample(filters, size, apply_batchnorm=True):
    """Downsampling block: Conv2D -> BatchNorm (optional) -> LeakyReLU."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """Upsampling block: Conv2DTranspose -> BatchNorm -> Dropout (optional) -> ReLU."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# -------------------------
# Generator Model (U-Net)
# -------------------------

def Generator():
    """
    Modified U-Net generator:
    - Encoder: Downsampling blocks with skip connections
    - Decoder: Upsampling blocks with skip connections and dropout in first 3 blocks
    """
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 256, 256, 64)
        downsample(128, 4),                        # (bs, 128, 128, 128)
        downsample(256, 4),                        # (bs, 64, 64, 256)
        downsample(512, 4),                        # (bs, 32, 32, 512)
        downsample(512, 4),                        # (bs, 16, 16, 512)
        downsample(512, 4),                        # (bs, 8, 8, 512)
        downsample(512, 4),                        # (bs, 4, 4, 512)
        downsample(512, 4),                        # (bs, 2, 2, 512)
        downsample(512, 4),                        # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),     # (bs, 2, 2, 512)
        upsample(512, 4, apply_dropout=True),     # (bs, 4, 4, 512)
        upsample(512, 4, apply_dropout=True),     # (bs, 8, 8, 512)
        upsample(512, 4, apply_dropout=True),     # (bs, 16, 16, 512)
        upsample(512, 4),                         # (bs, 32, 32, 512)
        upsample(256, 4),                         # (bs, 64, 64, 256)
        upsample(128, 4),                         # (bs, 128, 128, 128)
        upsample(64, 4),                          # (bs, 256, 256, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 512, 512, 3)

    x = inputs

    # Encoder downsampling with skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder upsampling + skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# Test generator output shape
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])


# -------------------------
# Loss Functions
# -------------------------

LAMBDA = 100  # Weight for L1 loss

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    """
    Generator loss combines adversarial loss and L1 loss.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Discriminator loss sums real and generated losses.
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# -------------------------
# Discriminator Model (PatchGAN)
# -------------------------

def Discriminator():
    """
    PatchGAN discriminator that classifies whether image patches are real or fake.
    Takes input image and target (real or generated) image concatenated.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # Concatenate along channels

    down1 = downsample(64, 4, False)(x)   # (bs, 256, 256, 64)
    down2 = downsample(128, 4)(down1)     # (bs, 128, 128, 128)
    down3 = downsample(256, 4)(down2)     # (bs, 64, 64, 256)
    down4 = downsample(512, 4)(down3)     # (bs, 32, 32, 512)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (bs, 34, 34, 512)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# Visual test of discriminator output
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()


# -------------------------
# Optimizers and Checkpoints
# -------------------------

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = pathMain + '\\training_checkpoints\\'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# -------------------------
# Image Generation Utility
# -------------------------

def generate_images(model, test_input, tar, counter):
    """
    Generate and save side-by-side comparison images of:
    input, ground truth, and generated output.
    """
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # scale [-1,1] -> [0,1]
        plt.axis('off')
    plt.savefig('epoch' + str(counter) + '.jpg')


# -------------------------
# Training Step and Loop
# -------------------------

EPOCHS = 500
import datetime

log_dir = "logs\\"
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)


@tf.function
def train_step(input_image, target, epoch):
    """
    Single training step:
    - Generate output with generator
    - Calculate discriminator outputs for real and generated images
    - Compute losses and gradients
    - Apply gradients using optimizers
    - Log losses to TensorBoard
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
    """
    Main training loop:
    - Iterates over epochs
    - Runs training steps for each batch
    - Generates and saves example outputs every 10 epochs
    - Saves model checkpoints every 100 epochs
    """
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        if (epoch + 1) % 10 == 0:
            for example_input, example_target in test_ds.take(1):
                generate_images(generator, example_input, example_target, epoch)

        print("Epoch: ", epoch)

        # Train batches
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time() - start))

    checkpoint.save(file_prefix=checkpoint_prefix)


# Uncomment to run training loop
# fit(train_dataset, EPOCHS, test_dataset)

print(checkpoint_dir)
print(tf.train.latest_checkpoint(checkpoint_dir))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run inference and save images for some test examples
cnt = 0
for inp, tar in test_dataset.take(5):
    cnt += 1
    generate_images(generator, inp, tar, cnt)
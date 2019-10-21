from __future__ import absolute_import, division, print_function, unicode_literals

import math
import tensorflow as tf
import tensorflow_datasets as tfds

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale_image(image, label):
    image = tf.dtypes.cast(image, tf.float32)
    image /= 255
    return image, label


# Create neural network model
def build_model():
    # Declare model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.9, decay=1e-6),
      metrics=['accuracy']
    )

    # Output summary and return
    model.summary()
    return model

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets buffer/batch sizes
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS
print("{} batch size".format(BATCH_SIZE))

# Define and load datasets
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
NUM_OF_TRAIN_SAMPLES = info.splits['train'].num_examples
NUM_OF_TEST_SAMPLES = info.splits['test'].num_examples
STEPS_PER_EPOCH = math.ceil(NUM_OF_TRAIN_SAMPLES//BATCH_SIZE)
print("{} samples in training dataset, {} samples in testing dataset, {} steps in one epoch".format(NUM_OF_TRAIN_SAMPLES, NUM_OF_TEST_SAMPLES, STEPS_PER_EPOCH))
dataset_train_raw = datasets['train']
dataset_test_raw = datasets['test']

# Prepare training/testing dataset
dataset_train = dataset_train_raw.map(scale_image).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
dataset_test = dataset_test_raw.map(scale_image).batch(BATCH_SIZE)

# Build and train the model as multi worker
with strategy.scope():
    model = build_model()
model.fit(dataset_train, epochs=5, steps_per_epoch=STEPS_PER_EPOCH)

# Show model summary, and evaluate it
model.summary()
eval_loss, eval_acc = model.evaluate(dataset_test)
print("\nEval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

# Save the model
model.save("model.h5")
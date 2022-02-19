import os
import shutil

import tensorflow as tf

def train(
    model,
    dataset,
    epochs=100,
    optimizer="adam",
    learning_rate=1e-5,
    loss="mse",
    total_steps=None,
    checkpoint_frequecy=10,
    learning_rate_func=None,
    mixed_precision=True,
    working_dir="training_dir",
    log_every_step=1,
    validation=False,
):
    """
    Trains the model of interest.

    :param model (tensorflow.keras.model): Model to be trained on
    :param dataset (tensorflow.data.Dataset): Dataset to use for training
    :returns model:
    """
    # Directory Configurations
    tensorboard_dir = os.path.join(
        working_dir,
        "tensorboard"
    )
    model_dir = os.path.join(
        working_dir,
        "saved-models"
    )
    checkpoint_dir = os.path.join(
        working_dir,
        "checkpoint"
    )
    if os.path.isdir(working_dir):
        input("Press Enter to continue...")
        shutil.rmtree(working_dir)
    else:
        os.mkdir(working_dir)
        os.mkdir(tensorboard_dir)
        os.mkdir(model_dir)
        os.mkdir(checkpoint_dir)

    # Tensorboard functions
    tb_summary_writer = tf.summary.create_file_writer(
        tensorboard_dir
    )

    # Loss functions
    if loss == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = loss

    # Optimizers
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Mixed precision
    if mixed_precision == True:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Checkpointing 
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, 
        directory=checkpoint_dir, 
        max_to_keep=5
    )

    step = 0 
    # Training Loop
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        for image, label in dataset:
            with tf.GradientTape() as tape:
                logits = model(image, training=True)
                loss = loss_fn(label, logits)
                if mixed_precision:
                    loss = optimizer.get_scaled_loss(loss)
            gradients = tape.gradient(
                loss, 
                model.trainable_variables
            )
            if mixed_precision:
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            step = step + 1

            # Logging and checkpoint
            if step % log_every_step == 0:
                tf.summary.scalar("Loss", loss, step=step)
                tf.summary.scalar("Learning rate", learning_rate, step=step)
            if step % checkpoint_frequecy == 0:
                checkpoint_manager.save()
            # Learning rate adjustment
            if learning_rate_func != None:
                updated_learning_rate = learning_rate_func(step, total_steps)
                optimizer.update_learning_rate(
                    updated_learning_rate
                )

            output_format = "Step: {} Loss: {} Learning rate: {}"
            print(
                output_format.format(
                    step,
                    loss,
                    learning_rate
                )
            )
        # Save the trained model
        tf.keras.models.save_model(model, model_dir)
    
    print("Training Complete")

if __name__ == "__main__":
    from dataset import KeypointsDataset
    from model import build_model

    # General Variables
    PATH_TO_DATASET = "dataset"
    PATH_TO_ANNOTATION_FILE = "dataset/ochuman.json"

    # Create the dataset
    dataset_gen = KeypointsDataset(
        dataset_dir=PATH_TO_DATASET,
        annotation_file=PATH_TO_ANNOTATION_FILE,
        create_keypoints_files=False
    )
    dataset = dataset_gen.create_dataset()

    # Build the model for training
    model = build_model()

    # Run training
    train(
        model,
        dataset
    )
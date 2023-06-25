# CarModel_Prediction

This repository contains code for fine-tuning the MobileNetV2 model using the Keras library. The code allows you to train the model on a custom dataset, evaluate its performance, and make predictions on new images.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.0 or higher)
- Keras (version 2.3.1 or higher)
- Matplotlib (version 3.2.1 or higher)
- NumPy (version 1.18.1 or higher)
- pickle (version 4.0 or higher)
- imutils (version 0.5.3 or higher)

## Dataset Preparation

1. Prepare your dataset by organizing it into a train directory and a validation directory. Each directory should contain subdirectories for each class, where the images of that class are stored.
2. Update the `train_dr` variable in the code to point to the directory containing the training images.
3. Update the `train_images` variable in the code to list the paths of all training images using the `list(paths.list_images(directory))` function.
4. Update the `num_classes` variable in the code to the number of classes in your dataset.

## Model Configuration

The code uses the MobileNetV2 model as the base model for fine-tuning. The MobileNetV2 model is pretrained on the ImageNet dataset and achieves good performance in various computer vision tasks.

The `get_mobilenetv2_full_tune_model_alpha_1_4_concatenated_regularised` function in the code creates the MobileNetV2 model with specific configurations. You can modify the function to adjust the model architecture, such as changing the alpha value, adding or removing layers, or modifying the regularization parameters.

## Training

1. Set the desired hyperparameters for training, such as batch size, learning rate, momentum, number of epochs, and optimizer.
2. Set the early stopping patience, reduce learning rate on plateau factor, and reduce learning rate on plateau patience.
3. Run the code to start training the model. The code will preprocess the images, create data generators, and train the model using the specified hyperparameters and callbacks.
4. The training progress will be displayed, and the best model based on the validation loss will be saved to the `best_model.h5` file.

## Evaluation

1. After training, the model will be evaluated on the validation data using the `evaluate_generator` function.
2. The validation loss, accuracy, and top-10 accuracy will be displayed.

## Prediction

1. Load the best model using the `load_model` function and provide the path to the `best_model.h5` file.
2. Load an image that you want to make predictions on. Update the `img_path` variable in the code to point to the image file.
3. Run the code to preprocess the image and make predictions using the loaded model.
4. The top 5 predictions with their corresponding class names and probabilities will be displayed.

Please note that the code provided is a template and may require modifications based on your specific use case. Ensure that you update the variables and functions as needed and handle any errors or exceptions that may occur during the execution of the code.
### Dataset Used
https://data.mendeley.com/datasets/hj3vvx5946/1

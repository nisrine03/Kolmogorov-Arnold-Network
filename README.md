Segmentation of Brain MRI Images with a Modified U-Net Model and KAN Layers (Kolmogorov Arnold Network)

General Description
This script implements a brain MRI image segmentation model using an enhanced U-Net model with KAN layers, a network based on B-splines for extracting nonlinear relationships. It processes MRI images and their corresponding masks to extract sub-volumes, normalizes them, and trains a U-Net model with classical convolutional layers and KAN layers.

Imported Packages

numpy: For numerical computations.
nibabel: For loading and manipulating MRI images in NIfTI format.
scipy.ndimage.zoom: For image processing, specifically used for zooming in on volumes.
matplotlib.pyplot: For visualizing images and plots.
pandas: For data manipulation and management in DataFrame format.
torch (PyTorch): Library for creating and training deep learning models.
timm.models.layers: For specific layers like DropPath and other operations.
scikit-learn: For splitting datasets into training and test sets.
Code Details

Image Preprocessing

combine_masks_to_single_label(mask_gray, mask_white, mask_csf): Combines three masks (gray matter, white matter, and cerebrospinal fluid) into a single mask with values representing the different tissue classes.
normalize_min_max(image): Normalizes the pixel values of an image between 0 and 1 using min-max scaling.
one_hot_encode(mask, num_classes): Converts a segmentation mask into one-hot encoding, creating a tensor with additional channels for each class.
Training Sub-Volume Generation
get_sub_volume(image, label, ...): Extracts sub-volumes of images and masks for training the model. This function avoids sub-volumes dominated by the background.
Loading MRI Images and Masks

BrainMRIDataset: A custom class for loading MRI images and their associated masks. It is used by DataLoaders for training and validation.
load_brain_mri_dataset(path_img, path_mask): Loads the paths of images and masks and creates a DataFrame that associates each image with its corresponding masks.
Segmentation Model - Modified U-Net with KAN

UKAN: A segmentation model based on U-Net with KAN layers, designed to capture nonlinear relationships through B-splines.
ConvLayer: A convolutional layer with normalization and ReLU activation for encoding.
D_ConvLayer: A convolutional layer for the decoding phase.
KANLayer: Implementation of a KAN layer that combines B-splines and convolutions to extract complex information.
Training and Evaluation

DiceCoefficient: A class for calculating the Dice coefficient, a metric used to evaluate the similarity between predicted masks and actual masks.
Training: The model is trained using the Adam optimizer and CrossEntropy loss function. Losses and Dice Coefficients are calculated at each epoch to monitor model performance.
Evaluation: The model is evaluated on a validation set, and both loss and Dice Coefficients are tracked.
Results Visualization

display_results: A function that displays a slice of an MRI image, the actual mask, and the predicted mask by the model for clear visualization of segmentation performance.
Summary of Key Points

Main Goal: Create a modified U-Net model for brain MRI image segmentation, with KAN layers to capture complex nonlinear relationships.
Methodology: Preprocessing of MRI images, normalization, sub-volume extraction, and training a U-Net model with both classical convolutional layers and KAN layers.
Evaluation: Use of the Dice coefficient to evaluate model predictions against true masks.
Results: Visualization of the model's predictions in the form of MRI slices, comparing them with the actual masks.
Visualization
The model generates graphs to track loss and Dice Coefficient during training. MRI images with both the actual and predicted masks are displayed to evaluate the segmentation quality.

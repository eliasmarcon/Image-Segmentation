<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <h1 align="center">Cityscapes Dataset Segmentation Project</h1>
  <p align="center">
    #TODO
    <br />
    ·
    <a href="https://github.com/eliasmarcon/Image-Segmentation/issues">Report Bug</a>
    ·
    <a href="https://github.com/eliasmarcon/Image-Segmentation/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#installation">Requirements</a></li>
      <ol>
          <li><a href="#using-python-with-requirementstxt">Using Python with requirements txt file</a></li>
          <li><a href="#using-conda-on-a-slurm-cluster">Using Conda on a SLURM Cluster</a></li>
      </ol>
    <li><a href="#dataset">Dataset</a></li>
      <ol>
        <li><a href="#data-augmentation">Data Augmentation</a></li>
      </ol>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#model-architectures">Model Architectures</a></li>
      <ol>
        <li><a href="#cnnbasic">CNNBasic</a></li>
        <li><a href="#resnet">ResNet</a></li>
        <li><a href="#vit-vision-transformer">ViT (Vision Transformer)</a></li>
      </ol>
    <li><a href="#training">Training</a></li>
      <ol>
        <li><a href="#locally">Locally</a></li>
        <li><a href="#slurm-cluster">Slurm Cluster</a></li>
        <li><a href="#tracking--logging">Tracking / Logging</a></li> 
      </ol>
    <li><a href="#testing-results">Testing Results</a></li>
      <ol>
        <li><a href="#results-cnn">Results CNN</a></li>
        <li><a href="#results-resnet">Results ResNet</a></li>
        <li><a href="#results-vit">Results ViT</a></li>
      </ol>
    <li><a href="#overall-results">Overall Results</a></li>
  </ol>
</details>

<br>

# Requirements

Before running the code locally or on a SLURM cluster, please ensure the necessary dependencies are installed. For detailed instructions on setting up the environment look at the sections below. In order to run the code and created models, refer to the [Training & Testing](#training) section.

## Using Python with requirements.txt

1. **Local Environment:**

   - Ensure Python (>=3.10) and pip are installed.

   - Clone the repository:

     ```bash
     git clone <repository_url>
     cd <repository_name>
     ```

   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Using Conda on a SLURM Cluster

1. **Setting Up Conda Environment:**

   - Connect to the SLURM cluster.

   - Install all requirements and other dependencies:

     ```bash
        sbatch slurm_cluster/setup_conda.sh
     ```

   - Ensure all provided data dependencies are available and accessible within your environment.

2. **Running the Code:**

   - in the `./slurm_cluster/run_cluster.sh` and `./slurm_cluster/run_cluster_sets.sh` file the created conda environment is activated, used and then after the job is done, deactivated.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Dataset

The Cityscapes dataset is a large-scale collection of high-quality, pixel-level annotated images of urban scenes from 50 different cities in Germany. It is widely used for training and evaluating models in semantic segmentation, instance segmentation, and scene parsing tasks. The dataset includes 5,000 finely annotated images and 20,000 coarsely annotated images, capturing various weather conditions, seasons, and urban environments.

This project focuses on 19 of the 30 object classes, including essential elements like cars, pedestrians, and roadways.

To learn more about the Cityscapes dataset, visit the [official dataset page](https://www.cityscapes-dataset.com/) and their [official repository page](https://github.com/mcordts/cityscapesScripts).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Data Augmentation

For experiments involving data augmentation, the following transformations are applied to the Cityscapes dataset to enhance model robustness and generalization:

- `Horizontal Flip`: Randomly flips images and target labels horizontally with a probability of 50%, helping the model learn invariance to horizontal changes.
- `Adjust Sharpness`: Randomly adjusts the sharpness of the image with a sharpness factor of 2 and a 50% probability, improving the model's ability to handle varying image clarity.
- `Color Jitter`: Applies random changes to the brightness, contrast, saturation, and hue of the images to increase the model's robustness to color variations.
- `Gaussian Blur`: Applies a Gaussian blur to the images with a kernel size of 3 and a sigma range of 0.1 to 2.0, aiding in noise reduction and model stability.
- `Resize`: Resizes both images and target labels to a specified size, using nearest-neighbor interpolation, ensuring consistency in input dimensions.
- `Normalize`: Normalizes the images using the mean and standard deviation values of [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225], respectively, aligning them with pre-trained model requirements.

When augmentation is not applied, the dataset is preprocessed with resizing and normalization only.

The code for these augmentations can be found in the `./src/dataset/utils_dataset.py` file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


# File Structure

The file structure of this repository is designed to keep things organized and accessible, allowing for easy navigation and efficient project management.

```
├── cityscapes_data/          # Contains all files related to the Cityscapes dataset.
│   
├── hyperparameter_sets/      # Stores configurations for different sets of hyperparameters used in experiments.
│   
├── saved_models/             # Includes files for models that have been trained and saved.
│   
├── slurm_cluster/            # Contains scripts for managing and running jobs on a SLURM cluster.
│   ├── setup_conda.sh        # Script to set up Conda environment on the cluster.
│   ├── run_cluster.sh        # Script to execute jobs on the cluster.
│   ├── run_cluster_sets.sh   # Script for running multiple job sets (one hyperparameter file) on the cluster.
│
├── src/                      # Source code for the project.
│   ├── dataset/              # Contains code for dataset creation and manipulation.
│   ├── logger/               # Includes logging utilities for tracking experiments.
│   ├── metrics/              # Contains code for evaluating model performance.
│   ├── models/               # Includes model architecture definitions.
│   ├── trainer_tester/       # Contains training and testing logic.
│   └── main.py               # The main entry point for executing the code.
│
├── clean_runs.csv            # Log of runs and results
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Model Architectures

This repository includes several model architectures implemented for image segmentation tasks. Below are brief descriptions of each model.


## UNet Architecture

The `UNet` is a fully convolutional neural network designed for semantic segmentation tasks. It features an encoder-decoder structure that enables it to capture both the high-level context and detailed spatial information from input images. Here's a breakdown of the architecture:

- `Encoder Path`:
    - The encoder consists of a series of downsampling blocks, each containing convolutional layers that reduce the spatial dimensions while increasing the depth of the feature maps.
    - These blocks progressively extract and compress the image features, creating a hierarchy of encoded representations.
    - The encoder outputs are stored at each level to be used later in the decoding process.

- `Bottleneck`:
    - The bottleneck connects the encoder and decoder paths, featuring a DoubleConv block with two convolutional layers, which further refines the feature maps at the deepest level of the network.

- `Decoder Path`:
    - The decoder mirrors the encoder with a series of upsampling blocks that progressively restore the spatial dimensions of the feature maps.
    - Each upsampling block combines the upsampled features with the corresponding encoder outputs through skip connections, enabling the network to retain and utilize the high-resolution information from the encoder.
    - The decoder gradually reconstructs the spatial structure, leading to more precise segmentation outputs.

- `Output Layer`:
    - The final output layer is a 1x1 convolution that maps the decoded features to the desired number of segmentation classes. This layer outputs a feature map where each pixel represents the class probabilities for that location.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## FCNResNet (Fully Convolutional Resnet)

The `FCNResNet` is a fully convolutional network (FCN) that integrates the powerful feature extraction capabilities of ResNet with a fully convolutional head for semantic segmentation tasks. Here's a breakdown of the architecture:

- `ResNet Backbone`:

    - The backbone of this model is based on the ResNet architecture, which is renowned for its deep residual learning capabilities.
    - It can be configured with either BasicBlock for shallower networks (e.g., ResNet18, ResNet34) or Bottleneck for deeper ones (e.g., ResNet50, ResNet101, ResNet152).
    - The backbone comprises four residual blocks that progressively downsample the input while extracting hierarchical features.

- `FCN Head`:

    - The FCN head takes the deep features from the ResNet backbone and processes them through a series of convolutional, batch normalization, and ReLU layers.
    - This head reduces the channels via a bottleneck convolution and outputs the class scores for each pixel in the image.
    - The head configuration varies depending on the type of ResNet block used: for BasicBlock, the head processes 512 channels, whereas for Bottleneck, it processes 2048 channels.

- `Upsampling`:

    - After feature extraction and classification, the output is upsampled using bilinear interpolation to match the original image size. This step ensures that the final segmentation map has the same dimensions as the input image, allowing for pixel-wise classification.

- `ResNet Backbone Details`

    - `Initial Convolutional Layer`:
        - The first layer is a 7x7 convolution with a stride of 2, followed by batch normalization and a ReLU activation. This layer is responsible for initial low-level feature extraction.
    
    - `Residual Blocks`:
        - The backbone includes four residual blocks, each consisting of several BasicBlock or Bottleneck units. These blocks are designed to learn residual functions, allowing the network to efficiently capture both low-level and high-level features.
        - Downsampling occurs at the beginning of each residual block (except the first one), which helps in reducing the spatial dimensions and increasing the receptive field.

- `FCN Head Details`
    - `Convolutional Layers`:
        - The FCN head begins with a 3x3 convolution that reduces the number of input channels by a factor of 4, followed by batch normalization and a ReLU activation.
        - A dropout layer (with a probability of 0.1) is applied to improve generalization.
        - Finally, a 1x1 convolution is used to generate the output feature map with the desired number of segmentation classes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## SegFormer

`SegFormer` is designed for efficient and accurate image segmentation, combining the strengths of Vision Transformers (ViTs) with lightweight design principles.

- `Key Components`
    
    - `MixVisionTransformer (MViT) Encoder`:
        - The encoder utilizes a multi-scale feature extraction mechanism through the MixVisionTransformer class.
        - It includes a series of OverlapPatchEmbed layers for hierarchical feature extraction, followed by Transformer blocks that leverage self-attention across the multi-scale features.
    
    - `SegFormerHead Decoder`:
        - The decoder reconstructs the full-resolution segmentation map from the multi-scale features provided by the encoder.
        - It uses an MLP-based design (SegFormerHead) to effectively fuse the features across different scales and output the final segmentation.

    - `Model Architecture`:
        - The SegFormer class integrates the MViT encoder and the SegFormerHead decoder, providing an end-to-end model for semantic segmentation tasks.

- `Code Structure`
    - `MixVisionTransformer`: Defines the encoder structure with multi-scale self-attention blocks.
    - `SegFormerHead`: Implements the MLP-based decoder for upsampling and prediction.
    - `SegFormer`: The main model class that integrates the encoder and decoder.


`References`
- Official SegFormer paper: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- Original source code: [NVIDIA/SegFormer GitHub Repository](https://github.com/NVlabs/SegFormer)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Training

To optimize the training of deep learning models, a thorough exploration of various hyperparameter combinations was conducted. The summary below highlights the hyperparameters tested, providing an initial insight into which models perform best and whether larger model sizes lead to improved performance. 

The first hyperparameters are these:

| Hyperparameter    | Values |
| ----------------- | -------------------------------------- | 
| Model             | <p> [`resnet_18`, `resnet_50`, <p> `segformer_small`, `segformer_base`, <p> `unet_small`, `unet_base`]          |
| Data Augmentation | [`False`, `True`]                      | 
| Learning Rate     | [`0.01`, `0.001`, `0.0001`, `0.00001`] | 
| Weight Decay      | [`0.01`, `0.001`, `0.0001`, `0.00001`] | 
| Gamma             | [`0.95`, `0.97`, `0.99`]               | 


These hyperparameters result in the training of `576 models`. The models can be trained either locally (with the normal `./src/main.py` file) or on a Slurm cluster using specific scripts: `./slurm_cluster/run_cluster.sh` and `./slurm_cluster/run_cluster_sets.sh`. These models were all trained on a `NVIDIA GeForce RTX 2080 Ti`.


The second (and final) hyperparameters are these:

| Hyperparameter    | Values |
| ----------------- | -------------------------------------- | 
| Model             | <p> [`resnet_18`, `resnet_50`, <p> `segformer_small`, `segformer_base`, <p> `unet_small`, `unet_base`]          |
| Data Augmentation | [`False`, `True`]                      | 
| Learning Rate     | [`0.01`, `0.001`, `0.0001`, `0.00001`] | 
| Weight Decay      | [`0.01`, `0.001`, `0.0001`, `0.00001`] | 
| Gamma             | [`0.95`, `0.97`, `0.99`]               | 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Locally

To run experiments locally, use the following command format:

```bash
# General Arguments
python src/main.py -d [DATASET_PATH]            # Optional
                   -s [SAVE_MODEL_PATH]         # Optional
                   -e [NUM_EPOCHS] 
                   -b [BATCH_SIZE] 
                   -m [MODEL_TYPE] 
                   -a [AUGMENTATION] 
                   -l [LEARNING_RATE] 
                   -w [WEIGHT_DECAY] 
                   -g [GAMMA] 
                   -f [VAL_FREQUENCY]           # Optional 
                   -p [EARLY_STOPPING_PATIENCE] # Optional

# Example
python src/main.py -d ./cityscapes_data -s ./saved_models -e 80 -b 16 -m resnet_50 -a -l 0.0001 -w 0.0001 -g 0.99 -f 1 -p 20
```

Breakdown of the Command:

| Argument   | Value               | Description                                                                      |
|:----------:|---------------------|----------------------------------------------------------------------------------|
| `-d`       | `./cityscapes_data` | Sets the path to the dataset to ./cityscapes_data (default: ./cityscapes_data).  |
| `-s`       | `./saved_models`	   | Sets the base path to save the model to ./saved_models (default: ./saved_models).|
| `-e`       | `80`	               | Specifies the number of epochs to train the model (default: 80).                 |
| `-b`       | `16`	               | Defines the batch size as 16 (default: 16).                                      |
| `-m`       | `resnet_50`	       | Specifies the model type as resnet_50 (default: resnet_50).                      |
| `-a`       | `(TRUE)`	           | Enables data augmentation (default: False).                                      |
| `-l`       | `0.0001`	           | Sets the learning rate to 0.0001 (default: 0.0001).                              |
| `-w`       | `0.0001`	           | Specifies the weight decay as 0.0001 (default: 0.0001).                          |
| `-g`       | `0.99`	             | Sets the gamma value for the learning rate scheduler to 0.99 (default: 0.99).    |
| `-f`       | `1`	               | Sets the validation frequency to 1 (default: 1).                                 |
| `-p`       | `20`	               | Sets the early stopping patience to 20 (default: 20).                            |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Slurm Cluster

To run experiments on a Slurm Cluster, use one of the following command formats:

### For Individual Hyperparameter Testing

Use `run_cluster.sh` to test individual custom hyperparameters:

```bash
sbatch run_cluster.sh [NUM_EPOCHS] [BATCH_SIZE] [MODEL_TYPE] [AUGMENTATION] [LEARNING_RATE] [WEIGHT_DECAY] [GAMMA]

sbatch run_cluster.sh 50 64 resnet True 0.001 0.0001 0.97
```

### For Batch Hyperparameter Testing

Use run_cluster_sets.sh to run predefined sets of hyperparameters:

```bash
sbatch run_cluster_sets.sh [FOLDER_TYPE] [FILE_NUMBER]

sbatch run_cluster_sets.sh first_tests 7
```

### Script Details

`./slurm_cluster/run_cluster.sh:` 
- Allows testing of individual custom hyperparameters. 
- Suitable for running a single experiment with user-defined settings. 

`./slurm_cluster/run_cluster_sets.sh:`
- Runs predefined sets of hyperparameters in batch mode.
- Takes two arguments: folder_type and file_number.
- Fetches hyperparameter sets from ./hyperparameter_sets/{folder_type}/hyperparameters_{file_number}.txt.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Tracking / Logging

Metrics and model training details are logged using the Weights and Biases (wandb) platform. Weights and Biases provides a suite of tools for experiment tracking, including real-time visualization of metrics, model performance, and system metrics. For more details on setting up and using wandb, refer to the [Weights and Biases documentation](https://docs.wandb.ai/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>






# Training


<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Training


<p align="right">(<a href="#readme-top">back to top</a>)</p>

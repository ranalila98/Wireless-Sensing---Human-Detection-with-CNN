# Wireless-Sensing---Human-Detection-with-CNN

This project utilizes wireless sensing to detect human presence using a Convolutional Neural Network (CNN) and Channel State Information (CSI) within a Wireless Local Area Network (WLAN). The CSI data, captured using Software-Defined Radio (SDR), is provided as pre-recorded data for this project.

This work aligns with WLAN sensing standardized by the IEEE® 802.11bf task group.

## Project Overview

The core of this project involves processing CSI data to generate visual representations, which are then fed into a CNN for human presence detection.

**Key Features:**

* **Wireless Sensing:** Leverages WLAN CSI for human detection.
* **Convolutional Neural Network (CNN):** Employs a CNN for robust image classification.
* **Software-Defined Radio (SDR):** Utilizes pre-recorded SDR data.
* **IEEE® 802.11bf Compliance:** Aligns with WLAN sensing standards.

## Visual Examples

![Example 1](https://github.com/user-attachments/assets/06890c06-f53d-4e06-aa47-d058515f21f8)

![Example 2](https://github.com/user-attachments/assets/b292ffa4-10a7-4624-89e7-6d42d1030707)

![Example 3](https://github.com/user-attachments/assets/dad62d98-ccd4-4f1f-8996-aa44012b655a)

## Dataset

The dataset used in this project is available at:

[https://utdallas.box.com/s/nlfb0utfmh06p8vptbqkodfop5ioyfic](https://utdallas.box.com/s/nlfb0utfmh06p8vptbqkodfop5ioyfic)

## Getting Started

Follow these steps to set up and run the project:

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/ranalila98/Wireless-Sensing---Human-Detection-with-CNN.git](https://github.com/ranalila98/Wireless-Sensing---Human-Detection-with-CNN.git)
    ```

2.  **Navigate to the Project Directory:**

    ```bash
    cd Wireless-Sensing---Human-Detection-with-CNN
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Images from CSI Data:**

    ```bash
    python image_generator.py
    ```

    * This script processes the pre-recorded CSI data and generates images.

5.  **Train and Run the CNN Model:**

    ```bash
    python CNN.py
    ```

    * This script trains the CNN model using the generated images and evaluates its performance.

## Files Description

* `image_generator.py`: Generates images from the CSI dataset.
* `CNN.py`: Implements and trains the CNN for human detection.
* `requirements.txt`: Lists the Python packages required for the project.
* `README.md`: This document.

# Lightweight Neural Network for ECG and EEG Anomaly Detection

## Overview
This project implements a lightweight neural network for detecting anomalies in ECG (Electrocardiogram) and EEG (Electroencephalogram) signals. The goal is to provide an efficient and accurate solution for real-time medical diagnostics, leveraging a streamlined model architecture suitable for deployment on devices with limited computational resources.

## Features
- **Lightweight Model**: Designed to be resource-efficient, making it suitable for deployment on edge devices.
- **High Accuracy**: Capable of accurately detecting anomalies in ECG and EEG signals.
- **Real-time Processing**: Optimized for real-time anomaly detection, ensuring timely diagnostics.
- **Scalability**: Can be easily scaled and adapted to other biomedical signal analysis tasks.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SnehaVPujari007/Lightweight_Neutral_Network_for-ECG-EEG-Anomaly_Detection.git
    cd lightweight-neural-network-ecg-eeg
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
1. **Prepare your dataset**:
    - Ensure that your ECG and EEG data are properly formatted and preprocessed.
    - Update the `config.py` file with the path to your dataset and any other necessary configurations.

2. **Train the model**:
    ```bash
    python train.py
    ```
    The training script will save the best-performing model to the `models/` directory.

### Evaluating the Model
1. **Run the evaluation script**:
    ```bash
    python evaluate.py
    ```
    This will output performance metrics and generate evaluation plots.

## Contributing
Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support, please open an issue in the repository or contact us at [dotsnehapujari555@gmail.com](mailto:dotsnehapujari555@gmail.com).

Thank you for using our lightweight neural network for ECG and EEG anomaly detection! We hope it aids in your research and development endeavors.







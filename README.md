<h1>Light Weight Neural Network for ECG and EEG Anomaly Detection </h1>
<h3>Overview</h3>
<p>This project implements a lightweight neural network for detecting anomalies in ECG (Electrocardiogram) and EEG (Electroencephalogram) signals. 
The goal is to provide an efficient and accurate solution for real-time medical diagnostics, leveraging a streamlined model architecture suitable for deployment on devices with limited computational resources.</p>

<h3>Features</h3>
<li>Lightweight Model: Designed to be resource-efficient, making it suitable for deployment on edge devices.</li>
<li>High Accuracy: Capable of accurately detecting anomalies in ECG and EEG signals.</li>
<li>Real-time Processing: Optimized for real-time anomaly detection, ensuring timely diagnostics.</li>
<li>Scalability: Can be easily scaled and adapted to other biomedical signal analysis tasks.</li>
<h3>Installation</h3>
<ul>To set up the project, follow these steps:</ul>
<ol type = "1">
<li>Clone the repository:</li> </br>
<ul>git clone https://github.com/SnehaVPujari007/Lightweight_Neutral_Network_for-ECG-EEG-Anomaly_Detection.git</ul>
<ul>cd lightweight-neural-network-ecg-eeg</ul> </br>
<li>Create a virtual environment:</li></br>
<ul>python -m venv venv</ul>
<ul>source venv/bin/activate  # On Windows use `venv\Scripts\activate`</ul> </br>
<li>Install the required dependencies:</li> </br>
<ul>pip install -r requirements.txt</ul>

</ol>
<h3>Usage</h3>
<h4>Training the Model</h4>
<ol type = "1">
<li>Prepare your dataset:</li>

<ol>Ensure that your ECG and EEG data are properly formatted and preprocessed. </ol>
<ol>Update the config.py file with the path to your dataset and any other necessary configurations.</ol>
<li>Train the model:</li>
<ul>python train.py</ul>
<ul>The training script will save the best-performing model to the models/ directory.</ul>
</ol>
<h2>Evaluating the Model</h2>

<li>Run the evaluation script:</li> </br>
<ul>python evaluate.py</ul>
<ul>This will output performance metrics and generate evaluation plots.</ul>



<h3>Contributing </h3>
<ul>Please follow these steps to contribute:</ul>

<li>Fork the repository. </li>
<li>Create a new branch (git checkout -b feature-branch).</li>
<li>Make your changes and commit them (git commit -m 'Add new feature').</li>
<li>Push to the branch (git push origin feature-branch).</li>
<li>Create a pull request.</li>
<h3>License</h3>
<p>This project is licensed under the MIT License. See the LICENSE file for details.</p>

<h3>Contact </h3>
<p>For questions or support, please open an issue in the repository or contact us at dotsnehapujari555@gmail.com.</p>

<b>Thank you for using our lightweight neural network for ECG and EEG anomaly detection! We hope it aids in your research and development endeavors. </b>







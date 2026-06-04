# handwriting-recognition
This project focuses on building a deep learning–based handwriting writer identification system that can identify who wrote a handwritten document based on image data.


✍️ Writer Identification from Handwriting – Deep Learning Project
📌 Project Overview

This project focuses on building a deep learning–based handwriting writer identification system that can identify who wrote a handwritten document based on image data.

The system is designed to support forensic analysis by classifying handwriting samples according to their writer ID, which is embedded in the filename of each image.

1️⃣ Shared Understanding of the Project (Common Ground)
🧠 Project in Simple Words

We are building a deep learning system that:

Takes a handwritten image as input

Analyzes handwriting patterns

Predicts which writer produced the handwriting

Each image belongs to one writer, and the writer’s identity is encoded in the first two characters of the filename.

Example:

01_9_999.png  → Writer ID = 01

🎯 System Requirements

The final system must:

Learn handwriting patterns from training images

Predict the writer for unseen test images

Run offline, with no GPU, on a Windows CPU

Output predictions and accuracy in a specific CSV format

📌 Important Rule:

Test data must never be used during training.

2️⃣ The Full Pipeline (Everyone Must Understand This)

This section represents the end-to-end flow of the project.
Every team member must understand all stages, even if they are responsible for only one part.

🔹 1. Input

A folder containing handwritten image files

Filenames contain the writer ID

Data is split into:

Training set

Test set

Hold-out test set (final evaluation)

🔹 2. Preprocessing

Each image undergoes the following steps:

Resize to a fixed dimension (e.g., 64 × 64)

Convert to grayscale

Normalize pixel values (scale to [0, 1])

Purpose:
Ensure consistent input format and reduce noise.

🔹 3. Label Extraction

Extract the first two characters of the filename

Convert them into numeric class labels

Use label encoding suitable for classification

🔹 4. Model Training

Use a Convolutional Neural Network (CNN)

Train only on the training dataset

Learn visual handwriting features (strokes, curves, spacing)

Output:

A trained model saved as model.keras (or equivalent)

🔹 5. Evaluation

Load the trained model

Predict labels for test images

Compare predicted labels with actual labels

Compute average accuracy

🔹 6. Deployment

Implemented using run.py

Must:

Load the saved model

Read test images

Generate predictions

Compute average accuracy

Save results to result.csv

3️⃣ Project Structure
project/
├── train/                 # Training images
├── test/                  # Test images
├── holdout_test/          # Final evaluation images
│
├── notebooks/             # Development & experimentation
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── deployment_testing.ipynb
│
├── train.py               # Final training script
├── run.py                 # Final deployment & evaluation script
├── model.keras            # Trained model
├── result.csv             # Prediction results
├── report.pdf             # 1-page project report
└── README.md              # Project documentation


📌 Only train.py, run.py, and the model file are used for final evaluation.

4️⃣ Development Workflow (Team Agreement)

Jupyter notebooks are used for:

Exploration

Debugging

Visualization

Experimentation

Python scripts (.py) are used for:

Final training (train.py)

Final testing and deployment (run.py)

Notebooks are for thinking and experimenting.
Scripts are for submission and grading.

5️⃣ Output Specifications
📄 Model File

Filename: model.keras

Saved after training

Loaded by run.py

📊 Result CSV (result.csv)

The output file must contain the following columns, in this order:

1. `filename` – the image filename (e.g., `01_001.png`)
2. `actual_label` – the ground-truth writer ID (e.g., `01`)
3. `predicted_label` – the predicted writer ID for that image (e.g., `03`)
6️⃣ Constraints & Rules (Must Follow)

✔ Deep learning models only (CNN, Neural Networks)
✔ No traditional ML models allowed
✔ Runs offline (no internet)
✔ Runs on CPU only (no GPU)
✔ Python scripts only for submission
✔ Test data must not be used in training

7️⃣ Final Note to Team Members

Every team member should be able to:

Explain the entire pipeline

Describe how labels are extracted

Explain the CNN architecture

Explain how accuracy is computed

Run train.py and run.py independently



🔰 Getting Started

This section ensures that any team member or evaluator can run the project smoothly on a local Windows machine without internet or GPU.

🖥️ System Requirements

OS: Windows 10 / 11

CPU only (no GPU required)

Python 3.9 – 3.11 (recommended)

Minimum RAM: 8 GB (16 GB preferred)

📦 1. Create & Activate Virtual Environment (Recommended)

Open Command Prompt or PowerShell in the project directory:

python -m venv venv


Activate the environment:

venv\Scripts\activate

📥 2. Install Required Packages
pip install numpy opencv-python tensorflow scikit-learn pandas matplotlib


⚠️ Make sure TensorFlow is the CPU version.

📂 3. Dataset Setup

Ensure the dataset folders are placed at the root of the project:

project/
├── train/
├── test/
├── holdout_test/

🚀 4. Train the Model

Run the training script:

python train.py


Expected output:

Training logs printed in the terminal

Trained model saved as:

model.keras

🧪 5. Run Evaluation & Deployment

Run the deployment script:

python run.py


Expected output:

Average accuracy printed in terminal

Prediction results saved as:

result.csv

✅ 6. Verify Output Files

Ensure the following files exist after execution:

model.keras

result.csv

If these files are generated correctly, the project is ready for submission.

👥 Roles & Responsibilities

To ensure efficiency, quality, and fairness, responsibilities are clearly divided while maintaining shared understanding.

👤 Student 1 — Data & Preprocessing Lead

Primary Focus: Data correctness and preprocessing pipeline

Responsibilities

Understand dataset structure and filename conventions

Implement image preprocessing:

Resize

Grayscale conversion

Normalization

Extract and encode class labels correctly

Ensure no data leakage between training and test sets

Contribute preprocessing explanation for report & video

👤 Student 2 — Model & Training Lead

Primary Focus: Model performance and training quality

Responsibilities

Design CNN architecture (CPU-efficient)

Implement train.py

Select appropriate:

Loss function

Optimizer

Batch size

Epochs

Train and tune the model

Save final trained model correctly

Explain model architecture in report & video

👤 Student 3 — Deployment & Evaluation Lead

Primary Focus: Compliance, deployment, and final presentation

Responsibilities

Implement run.py

Load trained model and test images

Generate predictions and compute accuracy

Save predictions in result.csv

Verify offline, CPU-only execution

Lead video presentation and 1-page report compilation

🔁 Shared Responsibilities

All team members must:

Review each other’s code

Understand the full pipeline

Be able to explain any part of the system during demo or viva

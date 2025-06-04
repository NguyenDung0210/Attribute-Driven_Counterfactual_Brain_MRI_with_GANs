# Counterfactual MRI Image Generation using Multi-Attribute GANs
This repository contains multiple Generative Adversarial Network (GAN) architectures adapted to generate **counterfactual MRI brain images** based on input attributes such as age and sex.

---

## 📁 Repository Structure
```
.
├── data/ # Dataset folder
│ ├── load_MRI_data.ipynb # Data loading notebook
│ ├── check_data.ipynb # Data checking/visualization
├── src/ # GAN models
│ ├── CounterSynth/
│ ├── CycleGAN/
│ ├── StyleGAN/
│ └── DCGAN/
├── .gitignore
├── requirements.txt
└── README.md
```

## 📦 Requirements
Before running the code, install the required Python packages:

```bash
pip install -r requirements.txt
```

## 📂 Dataset Structure
Your dataset should follow the structure below:
```
📂 data
 ├── participants.xlsx
 └── sub-BrainAgeXXXXX/
     └── anat/
         └── sub-BrainAgeXXXXX_T1w.nii.gz
```
- participants.xlsx must contain the following columns:
  - subject_id
  - subject_age
  - subject_sex
  - (any other relevant metadata)

## 🚀 How to Run
### 1. Navigate to the GAN architecture you want to use inside the src/ folder. Available options:
- CounterSynth/
- CycleGAN/
- StyleGAN/
- DCGAN/

### 2. Open the corresponding .ipynb notebook in that folder.

### 3. In the notebook:
- Run the cell that loads and prepares the dataset (look for a class like BrainMRIDataset)
- Run the model definition cells for Generator and Discriminator
- Run the training loop setup

### 4. Continue executing the cells until the main() function appears.

### 5. Modify the arguments inside the main() call to:
- Set the correct path to your data/ folder
- Adjust training parameters as needed (e.g., batch size, epochs)

### 6. Run the main() function to start training.

## 📌 Notes
- All notebooks are self-contained. Simply running all cells in order up to and including main() should be enough to reproduce training.
- For data loading, you can also explore data/load_MRI_data.ipynb and data/check_data.ipynb.
- The dataset used in this research is internal and cannot be publicly shared due to privacy and data protection policies.
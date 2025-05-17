import os
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    """
    Dataset cho Brain MRI với thông tin tuổi và giới tính
    """
    def __init__(self, data_dir, participants_file, transform=None):
        """
        Khởi tạo Dataset
        
        Parameters:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            participants_file (str): Đường dẫn đến file Excel chứa thông tin người tham gia
            transform (callable, optional): Các phép biến đổi tùy chọn
        """
        self.data_dir = data_dir
        self.transform = transform
        self.participants_df = pd.read_excel(participants_file)
        
        # Chuẩn hóa giới tính thành giá trị số
        self.participants_df['gender_code'] = self.participants_df['subject_sex'].map({'m': 0, 'f': 1})
        
        # Chuẩn hóa tuổi (min-max scaling)
        self.min_age = self.participants_df['subject_age'].min()
        self.max_age = self.participants_df['subject_age'].max()
        self.participants_df['age_normalized'] = (self.participants_df['subject_age'] - self.min_age) / (self.max_age - self.min_age)
        
        valid_subjects = []
        for _, row in self.participants_df.iterrows():
            subject_id = row['subject_id']
            file_path = os.path.join(data_dir, subject_id, 'anat', f"{subject_id}_T1w.nii.gz")
            if os.path.exists(file_path):
                valid_subjects.append(subject_id)
        
        self.participants_df = self.participants_df[self.participants_df['subject_id'].isin(valid_subjects)]
        print(f"Tìm thấy {len(self.participants_df)} mẫu có dữ liệu MRI hợp lệ")
    
    def __len__(self):
        return len(self.participants_df)
    
    def __getitem__(self, idx):
        subject_info = self.participants_df.iloc[idx]
        subject_id = subject_info['subject_id']
        
        file_path = os.path.join(self.data_dir, subject_id, 'anat', f"{subject_id}_T1w.nii.gz")

        img = nib.load(file_path)
        img_data = img.get_fdata()

        axial_slice = img_data[:, :, img_data.shape[2]//2]
        sagittal_slice = img_data[img_data.shape[0]//2, :, :]
        coronal_slice = img_data[:, img_data.shape[1]//2, :]
        
        axial_slice = torch.from_numpy(axial_slice).float()
        sagittal_slice = torch.from_numpy(sagittal_slice).float()
        coronal_slice = torch.from_numpy(coronal_slice).float()
        
        # Min-max normalization
        axial_slice = (axial_slice - axial_slice.min()) / (axial_slice.max() - axial_slice.min() + 1e-8)
        sagittal_slice = (sagittal_slice - sagittal_slice.min()) / (sagittal_slice.max() - sagittal_slice.min() + 1e-8)
        coronal_slice = (coronal_slice - coronal_slice.min()) / (coronal_slice.max() - coronal_slice.min() + 1e-8)
        
        # Channel dimension
        axial_slice = axial_slice.unsqueeze(0)      # [1, H, W]
        sagittal_slice = sagittal_slice.unsqueeze(0)  # [1, H, W]
        coronal_slice = coronal_slice.unsqueeze(0)   # [1, H, W]
        
        age = torch.tensor(subject_info['age_normalized'], dtype=torch.float32)
        gender = torch.tensor(subject_info['gender_code'], dtype=torch.float32)
        
        slices = {
            'axial': axial_slice,
            'sagittal': sagittal_slice,
            'coronal': coronal_slice
        }
        
        condition = torch.tensor([age, gender], dtype=torch.float32)
        
        return slices, condition
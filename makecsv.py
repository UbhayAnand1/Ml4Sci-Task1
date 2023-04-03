import os
import csv

data_dir = 'C:/Users/abhay/OneDrive/Desktop/Lens_classification/dataset_preprocessed'
csv_file = 'labels.csv'
image_paths = []
labels = []
for label, class_name in enumerate(['no', 'sphere', 'vort']):
    class_dir = os.path.join(data_dir, 'test', class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_name, image_name)  # only store image name in CSV file
        image_paths.append(image_path)
        labels.append(label)  # use integer labels for better compatibility with PyTorch

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    for image_path, label in zip(image_paths, labels):
        writer.writerow([image_path, label])

# ransomware-detection-prototype
Prototype lightweight ransomware detection agent created for Cyber Security MSc

Two datasets are required but not included due to file size

**Static - Kaggle Ransomware Detection Dataset**

Download from: https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set
and save as: `datasets/static_ransom_data.csv`


**Dynamic — EldeRAN**

Download from: https://github.com/rissgrouphub/ransomwaredataset2016
and save as: `datasets/dynamic_ransom_data.csv`

Note: this file has no header row; column 0 is ID, column 1 is the Ransomware label,
column 2 is Ransomware_Family, and columns 3+ are features.
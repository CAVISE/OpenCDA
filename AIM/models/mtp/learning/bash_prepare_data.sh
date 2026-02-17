python3 -m AIM.models.mtp.learning.dataset_generate_csv --split train --num_seconds 5000 --create_new_vehicle_prob 0.08
python3 -m AIM.models.mtp.learning.dataset_generate_csv --split val --num_seconds 5000 --create_new_vehicle_prob 0.08
python3 -m AIM.models.mtp.learning.dataset_preprocess --csv_folder csv/train --pkl_folder csv/train_pre
python3 -m AIM.models.mtp.learning.dataset_preprocess --csv_folder csv/val --pkl_folder csv/val_pre
python3 -m AIM.models.mtp.learning.vizualize_data --after_preprocessing

DIR="$(dirname "$(realpath "$0")")"
cd $DIR

python3 dataset_generate_csv.py --split train --num_seconds 5000 --create_new_vehicle_prob 0.065
python3 dataset_generate_csv.py --split val --num_seconds 5000 --create_new_vehicle_prob 0.065
python3 dataset_preprocess.py --csv_folder csv/train --pkl_folder csv/train_pre
python3 dataset_preprocess.py --csv_folder csv/val --pkl_folder csv/val_pre
python3 vizualize_data.py --after_preprocess

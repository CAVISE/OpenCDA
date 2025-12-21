DIR="$(dirname "$(realpath "$0")")"
cd $DIR

python3 dataset_generate_csv.py --split train
python3 dataset_generate_csv.py --split val
python3 dataset_preprocess.py --csv_folder csv/train --pkl_folder csv/train_pre 
python3 dataset_preprocess.py --csv_folder csv/val --pkl_folder csv/val_pre 

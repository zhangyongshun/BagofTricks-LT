import json, os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--input_path",
        help="input train/test splitting files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--image_path",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args

def convert(input_path, image_root):
    train = open(os.path.join(input_path, 'ImageNet_LT_train.txt')).readlines()
    valid = open(os.path.join(input_path, 'ImageNet_LT_test.txt')).readlines()
    train_annos = []
    valid_annos = []
    print("Converting file {} ...".format(os.path.join(input_path, 'ImageNet_LT_train.txt')))
    idx = 0
    for info in tqdm(train):
        image, category_id = info.strip().split(' ')
        train_annos.append({"image_id": idx,
                          "category_id": int(category_id),
                          "fpath": os.path.join(image_root, image)
                            })
        idx += 1
    print("Converting file {} ...".format(os.path.join(input_path, 'ImageNet_LT_test.txt')))
    idx = 0
    for info in tqdm(valid):
        image, category_id = info.strip().split(' ')
        valid_annos.append({"image_id": idx,
                            "category_id": int(category_id),
                            "fpath": os.path.join(image_root, image)
                            })
        idx += 1
    num_classes = 1000
    return {"annotations": train_annos, "num_classes": num_classes}, {'annotations': valid_annos, "num_classes": num_classes}

if __name__ == "__main__":
    args = parse_args()
    train_annos, valid_annos = convert(args.input_path, args.image_path)
    print("Converted, Saveing converted file to {}".format(args.output_path))
    with open(os.path.join(args.output_path, 'ImageNet_LT_train.json'), "w") as f:
        json.dump(train_annos, f)
    with open(os.path.join(args.output_path, 'ImageNet_LT_val.json'), "w") as f:
        json.dump(valid_annos, f)
import tensorflow as tf
import cv2, os, json
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--input_path",
        help="input train/test splitting files",
        required=True,
        type=str,
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

def read_and_decode(filename_queue):
    """Parses a single tf.Example into image and label tensors."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image"], tf.uint8)
    image.set_shape([3*32*32])
    label = tf.cast(features["label"], tf.int32)
    return image, label


def convert_from_tfrecords(data_root, dir_name, num_class, mode, output_path, json_file_prefix):
    if mode == 'valid':
        tfrecord_path = os.path.join(data_root, dir_name, 'eval.tfrecords')
    else:
        tfrecord_path = os.path.join(data_root, dir_name, 'train.tfrecords')
    filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=False, num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    image, label = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    annotations = []
    try:
        step = 0
        while not coord.should_stop():
            images, labels = sess.run([image, label])
            images = cv2.cvtColor(images.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            im_path = os.path.join(output_path, json_file_prefix, 'images', str(labels))
            if not os.path.exists(im_path):
                os.makedirs(im_path)
            save_path = os.path.join(im_path, '{}_{}.jpg'.format(mode, step))
            cv2.imwrite(save_path, images)
            annotations.append({'fpath': save_path, 'image_id': step, 'category_id':int(labels)})
            step += 1
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()

    with open(os.path.join(output_path, json_file_prefix, json_file_prefix+'_{}.json'.format(mode)), 'w') as f:
        json.dump({'annotations': annotations, 'num_classes': num_class}, f)

    print('Json has been saved to', os.path.join(output_path, json_file_prefix, json_file_prefix+'_{}.json'.format(mode)))

if __name__ == '__main__':
    modes = ['train', 'valid']
    args = parse_args()

    cifar10_im50 = {'dir': 'cifar-10-data-im-0.02', 'json': 'cifar10_imbalance50', 'class': 10}
    cifar10_im100 = {'dir': 'cifar-10-data-im-0.01', 'json': 'cifar10_imbalance100', 'class':10}
    cifar100_im50 = {'dir': 'cifar-100-data-im-0.02', 'json': 'cifar100_imbalance50', 'class':100}
    cifar100_im100 = {'dir': 'cifar-100-data-im-0.01', 'json': 'cifar100_imbalance100', 'class': 100}

    for m in modes:
        convert_from_tfrecords(
            args.input_path, cifar10_im50['dir'],
            cifar10_im50['class'], m, args.output_path,
            cifar10_im50['json']
        )
        convert_from_tfrecords(
            args.input_path, cifar10_im100['dir'],
            cifar10_im100['class'], m, args.output_path,
            cifar10_im100['json']
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im100['dir'],
            cifar100_im100['class'], m, args.output_path,
            cifar100_im100['json']
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im50['dir'],
            cifar100_im50['class'], m, args.output_path,
            cifar100_im50['json']
        )



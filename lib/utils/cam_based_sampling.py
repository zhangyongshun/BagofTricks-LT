import numpy as np
import cv2
import copy
import os
from PIL import Image, ImageEnhance



def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    for anno in annotations:
        category_id = anno['category_id']
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list

def cam_based_sampling(dataset, model, cfg):
    annotations = dataset.get_annotations()
    num_classes = dataset.get_num_classes()
    #reset networks' feature_blobs
    model.cam_params_reset()

    generate_index = 0

    num_list, cat_list = get_category_list(annotations, num_classes)

    if not os.path.isdir(cfg.DATASET.CAM_DATA_SAVE_PATH):
        os.makedirs(cfg.DATASET.CAM_DATA_SAVE_PATH)

    print('*+*'*20)
    print('\n')
    print('CAM-based sampling starting ...')
    print('number of classes: {}, images in smallest class: {}, images in largest class: {}'.format(len(num_list), max(num_list), min(num_list)))

    num_absent_list = [max(num_list)+min(num_list) - num for num in num_list]

    if 'CIFAR' not in cfg.DATASET.DATASET:

        '''
        the threshold used in CAM-based-sampling
        '''

        print('using cam sampling threshold')
        print('threshold is: {}'.format(cfg.DATASET.CAM_NUMBER_THRES ))
        print('\n')

        for i in range(len(num_list)):
            if num_list[i]>=cfg.DATASET.CAM_NUMBER_THRES :
                num_absent_list[i] = 0
            elif num_absent_list[i] > cfg.DATASET.CAM_NUMBER_THRES :
                num_absent_list[i] = cfg.DATASET.CAM_NUMBER_THRES

    print('-----------------------------')
    print('Images need to be sampled: ', sum(num_absent_list))
    print('Images in original dataset: ', sum(num_list))
    print('Images in balanced dataset (original dataset + sampled images): ', sum(num_list)+sum(num_absent_list))
    print('-----------------------------')

    label_list = [int(d['category_id']) for d in dataset.data]
    label_array = np.array(label_list)
    label_index_gallery = list()
    for i in range(len(num_list)):
        label_index_gallery.append(np.argwhere(label_array==i).squeeze().flatten())

    cnt_rotate = 0
    cnt_translate = 0
    cnt_flip = 0
    cnt = 0

    cam_generation_data = []

    for i in range(len(num_absent_list)):
        for j in range(num_absent_list[i]):
            cnt += 1
            if cnt % 1000 == 0:
                print('We have generated {} images, the left images are {}'.format(cnt, sum(num_absent_list)-cnt))

            new_image_index = np.random.choice(label_index_gallery[i])
            origin_img = dataset._get_image(dataset.data[new_image_index])

            h, w, _ = origin_img.shape
            cam_groundtruth = model.get_CAM_with_groundtruth([new_image_index], dataset, label_list, (w, h))
            cam_groundtruth_mean = np.mean(cam_groundtruth[0])*3/4.0
            ignored_index = np.where(cam_groundtruth[0]<cam_groundtruth_mean)

            choice = np.random.randint(0, 3)
            if choice==0: ## rotate and scale
                cnt_rotate += 1
                cam_groundtruth_image = cam_groundtruth[0]
                cam_groundtruth_image[ignored_index] = 0
                h_min = -1
                h_max = -1
                w_min = -1
                w_max = -1
                for h_i in range(h):
                    if np.max(cam_groundtruth_image[h_i]):
                        if h_min!=-1:
                            h_max = h_i
                            break
                        else:
                            h_min = h_i
                for w_i in range(w):
                    if np.max(cam_groundtruth_image[:, w_i]):
                        if w_min!=-1:
                            w_max = w_i
                            break
                        else:
                            w_min = w_i
                rotate = np.random.randint(-45, 45)
                scale = np.random.randint(80, 120) / 100.
                M = cv2.getRotationMatrix2D(((w_min+w_max)/2, (h_min+h_max)/2), rotate, scale)
                rotate_and_scale_origin_image = cv2.warpAffine(origin_img, M, (w, h))
                rotate_and_scale_cam_image = cv2.warpAffine(cam_groundtruth_image, M, (w,h))
                rotate_and_scale_preserve_index = np.where(rotate_and_scale_cam_image)
                rotate_and_scale_origin_image_backup = copy.deepcopy(rotate_and_scale_origin_image)
                rotate_and_scale_origin_image[ignored_index] = 0
                rotate_and_scale_origin_image[rotate_and_scale_preserve_index] = rotate_and_scale_origin_image_backup[rotate_and_scale_preserve_index]
                origin_ignored_index = np.where(rotate_and_scale_origin_image)
                origin_img[origin_ignored_index] = 0
                final_img = origin_img + rotate_and_scale_origin_image
            elif choice==1: #translate
                cnt_translate += 1
                cam_groundtruth_image = cam_groundtruth[0]
                cam_groundtruth_image[ignored_index] = 0
                h_min = -1
                h_max = -1
                w_min = -1
                w_max = -1
                for h_i in range(h):
                    if np.max(cam_groundtruth_image[h_i]):
                        if h_min!=-1:
                            h_max = h_i
                            break
                        else:
                            h_min = h_i
                for w_i in range(w):
                    if np.max(cam_groundtruth_image[:, w_i]):
                        if w_min!=-1:
                            w_max = w_i
                            break
                        else:
                            w_min = w_i
                w_shift = np.random.randint(-1*w_min, w-w_max)
                h_shift = np.random.randint(-1*h_min, h-h_max)
                M = np.float32([[1,0,w_shift],[0,1,h_shift]])
                translate_cam_image = cv2.warpAffine(cam_groundtruth_image, M, (w,h))
                translate_origin_image = cv2.warpAffine(origin_img, M, (w, h))
                translate_preserve_index = np.where(translate_cam_image)
                translate_origin_image_backup = copy.deepcopy(translate_origin_image)
                translate_origin_image[ignored_index] = 0
                translate_origin_image[translate_preserve_index] = translate_origin_image_backup[translate_preserve_index]
                origin_ignored_index = np.where(translate_origin_image)
                origin_img[origin_ignored_index] = 0
                final_img = origin_img + translate_origin_image
            else:  ##horizontal filp
                cnt_flip += 1
                horizontal_image = cv2.flip(origin_img, 1)
                horizontal_cam_image = cv2.flip(cam_groundtruth[0], 1)
                horizontal_cam_image_mean = np.mean(horizontal_cam_image)*3/4.0
                horizontal_preserve_index = np.where(horizontal_cam_image>=horizontal_cam_image_mean)
                horizontal_image_backup = copy.deepcopy(horizontal_image)
                horizontal_image[ignored_index] = 0
                horizontal_image[horizontal_preserve_index] = horizontal_image_backup[horizontal_preserve_index]
                origin_ignored_index = np.where(horizontal_image)
                origin_img[origin_ignored_index] = 0
                final_img = origin_img + horizontal_image
            fpath = os.path.join(cfg.DATASET.CAM_DATA_SAVE_PATH, 'label_'+str(label_list[new_image_index])+'_generate_index_'+str(generate_index)+'.jpg')
            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            h, w, c = final_img.shape
            cv2.imwrite(fpath, final_img)
            cam_generation_data.append({
                'fpath': fpath,
                'im_height': h,
                'im_width': w,
                'category_id': label_list[new_image_index],
                'im_shape': (h, w, c)
            })
            generate_index += 1
            label_list.append(label_list[new_image_index])

    import json
    json.dump(cam_generation_data, open(cfg.DATASET.CAM_DATA_JSON_SAVE_PATH , 'w'))

    print('The sampled images have been save to ', cfg.DATASET.CAM_DATA_SAVE_PATH)
    print('The json file of balanced dataset has been saved to ', cfg.DATASET.CAM_DATA_JSON_SAVE_PATH)
    print('In cam_sampling, translated: {}, flips: {}, rotated and scaled: {}'.format(cnt_translate, cnt_flip, cnt_rotate))

    print('\n')
    print('*+*'*20)





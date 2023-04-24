import cv2
import torch
import json
import os
import random

from local_config import FLIR_DATASET_PATH


if __name__ == "__main__":
    full_data_path = os.path.join(FLIR_DATASET_PATH, "video_thermal_test")
    index_path = os.path.join(full_data_path, "index.json")
    with open(index_path, "r") as f:
        data_index = json.load(f)
    coco_path = os.path.join(full_data_path, "coco.json")
    with open(coco_path, "r") as f:
        data_coco = json.load(f)

    frame_dict = {an['id']: an for an in data_coco['images']}
    copy_frame_dict = frame_dict.copy()

    interest_categories = [1,  # person
                           3,  # car
                           10,  # light
                           12  # sign
                           ]

    annotations_to_consider = {
        1: [],  # person
        3: [],  # car
        10: [],  # light
        12: []  # sign
    }
    extend_bbox = 0.4

    for annotation in data_coco['annotations']:
        catgory_id = annotation['category_id']

        image_id = annotation['image_id']
        image_an = frame_dict[image_id]

        if catgory_id not in interest_categories:
            continue

        bbox = annotation['bbox']
        size = (bbox[2], bbox[3])
        diag = (bbox[2]**2 + bbox[3]**2)**0.5
        if diag < 16:
            continue

        if 'occluded' in annotation['extra_info']:
            if annotation['extra_info']['occluded'] != 'no_(fully_visible)':
                continue
        else:
            print("no occluded info")

        if 'hours' not in image_an['extra_info']:
            print("no hours info")
            continue

        if image_an['extra_info']['hours'] == 'nigth':
            continue

        max_side = max(bbox[2], bbox[3])
        max_side_e = max_side * (1. + extend_bbox)

        ebox = [
            bbox[0] + bbox[2] / 2 - max_side_e / 2,
            bbox[1] + bbox[3] / 2 - max_side_e / 2,
            max_side_e,
            max_side_e
        ]

        if ebox[0] < 0 or \
                ebox[1] < 0 or \
                ebox[0] + ebox[2] >= image_an['width'] or \
                ebox[1] + ebox[3] >= image_an['height']:
            continue

        annotations_to_consider[catgory_id].append(annotation)

    # balance annotations_to_consider
    min_len = min([len(annotations_to_consider[catgory_id])
                  for catgory_id in interest_categories])
    for catgory_id in interest_categories:
        # sample randomly for each category_id
        annotations_to_consider[catgory_id] = random.sample(
            annotations_to_consider[catgory_id], min_len)

    print("annotations_to_consider: ", [
          len(annotations_to_consider[catgory_id]) for catgory_id in interest_categories])

    # save bbox crops into folders
    for catgory_id in interest_categories:
        os.makedirs(os.path.join(full_data_path, 'crops',
                    repr(catgory_id)), exist_ok=True)
        for ind, annotation in enumerate(annotations_to_consider[catgory_id]):

            image_id = annotation['image_id']
            image_an = frame_dict[image_id]
            assert image_an == copy_frame_dict[image_id]
            image = cv2.imread(os.path.join(full_data_path, image_an['file_name']))

            bbox = annotation['bbox']
            size = (bbox[2], bbox[3])
            max_side = max(bbox[2], bbox[3])
            max_side_e = max_side * (1 + extend_bbox)
            # ebox = [bbox[0] - extend_bbox * size[0],
            #         bbox[1] - extend_bbox * size[1],
            #         bbox[2] + 2 * extend_bbox * size[0],
            #         bbox[3] + 2 * extend_bbox * size[1]]
            ebox = [
                bbox[0] + bbox[2] / 2 - max_side_e / 2,
                bbox[1] + bbox[3] / 2 - max_side_e / 2,
                max_side_e,
                max_side_e
            ]
            if ebox[0] < 0 or \
                    ebox[1] < 0 or \
                    ebox[0] + ebox[2] >= image_an['width'] or \
                    ebox[1] + ebox[3] >= image_an['height']:

                print(ebox)
                print('skipping ebx Oo')
                continue

            assert ebox[0] >= 0 and ebox[1] >= 0 and ebox[0] + \
                ebox[2] < image_an['width'] and ebox[1] + \
                ebox[3] < image_an['height']
            crop = image[int(ebox[1]):int(ebox[1]+ebox[3]),
                         int(ebox[0]): int(ebox[0]+ebox[2])]

            out_path = os.path.join(full_data_path, 'crops',
                                    repr(catgory_id),
                                    str(f'{ind:05}') + ".JPEG")
            print(f'{ind}. save to: ', out_path)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(out_path), gray_crop, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 100])

import cv2
import torch
import json
import os
import random
import argparse
from local_config import FLIR_DATASET_PATH
from script_manager.func.add_needed_args import smart_parse_args
from collections import Counter

def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]


parser = argparse.ArgumentParser(description='Arpgparse')
# dycs parameters
parser.add_argument('--output', default=None, type=str, help='output path')
parser.add_argument('--hours', default="[]", type=str2list, 
                    help='hours of the day that we accept (e.g. [night, day, dawn/dusk])')

parser.add_argument('--diag', default=16, type=int, help='minimal diagonal size')
parser.add_argument('--extend_bbox', default=0.4, type=float, help='extend bbox by this ratio')
parser.add_argument('--flir_subfolder', default="video_thermal_test", type=str, help='flir subfolder name')
parser.add_argument('--output_folder', default="crops", type=str, help='subfolder to save crops in flir_subfolder')

if __name__ == "__main__":

    args = smart_parse_args(parser)

    full_data_path = os.path.join(FLIR_DATASET_PATH, args.flir_subfolder)
    index_path = os.path.join(full_data_path, "index.json")
    with open(index_path, "r") as f:
        data_index = json.load(f)
    coco_path = os.path.join(full_data_path, "coco.json")
    with open(coco_path, "r") as f:
        data_coco = json.load(f)

    frame_dict = {an['id']: an for an in data_coco['images']}
    copy_frame_dict = frame_dict.copy()


    interest_categories = [
                   1,  # person
                   3,  # car
                   10,  # light
                   12  # sign
               ]

    name_map = {1: "person",
                3: "car",
                10: "light",
                12: "sign"}

    annotations_to_consider = {
        1: [],  # person
        3: [],  # car
        10: [], # light
        12: []  # sign
    }

    # save text to file
    text1 =  repr(Counter([v['extra_info']['hours'] for k,v in frame_dict.items() if 'hours' in v['extra_info']]))
    text2 = repr(Counter([v['extra_info']['occluded'] for v in data_coco['annotations'] if 'occluded' in v['extra_info']]))
    text = text1 + '\n' + repr(text2)
    with open(os.path.join(args.output, "flir.txt"), "w") as f:
        f.write(text)




    # images_rgb_train
    # Counter({'no_(fully_visible)': 88333,
    #          '1%_-_70%_occluded_(partially_occluded)': 71918,
    #          '70%_-_90%_occluded_(difficult_to_see)': 5836})
    # Counter({'night': 3658, 'day': 6191, 'dawn/dusk': 417})


    extend_bbox = args.extend_bbox

    for annotation in data_coco['annotations']:
        catgory_id = annotation['category_id']

        image_id = annotation['image_id']
        image_an = frame_dict[image_id]

        if catgory_id not in interest_categories:
            continue

        bbox = annotation['bbox']
        size = (bbox[2], bbox[3])
        diag = (bbox[2]**2 + bbox[3]**2)**0.5
        if diag < args.diag:
            continue

        if 'occluded' in annotation['extra_info']:
            if annotation['extra_info']['occluded'] != 'no_(fully_visible)':
                continue
        else:
            print("no occluded info")
            continue

        if 'hours' not in image_an['extra_info'] and 'all' not in args.hours:
            print("no hours info")
            continue

        if 'all' not in args.hours and image_an['extra_info']['hours'] not in args.hours:
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
        output_folder = os.path.join(args.output, args.flir_subfolder,
                                        name_map[catgory_id])
        os.makedirs(output_folder, exist_ok=False)
        for ind, annotation in enumerate(annotations_to_consider[catgory_id]):

            image_id=annotation['image_id']
            image_an=frame_dict[image_id]
            assert image_an == copy_frame_dict[image_id]
            image=cv2.imread(os.path.join(
                full_data_path, image_an['file_name']))

            bbox=annotation['bbox']
            size=(bbox[2], bbox[3])
            max_side=max(bbox[2], bbox[3])
            max_side_e=max_side * (1 + extend_bbox)
            # ebox = [bbox[0] - extend_bbox * size[0],
            #         bbox[1] - extend_bbox * size[1],
            #         bbox[2] + 2 * extend_bbox * size[0],
            #         bbox[3] + 2 * extend_bbox * size[1]]
            ebox=[
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
            crop=image[int(ebox[1]):int(ebox[1]+ebox[3]),
                         int(ebox[0]): int(ebox[0]+ebox[2])]

            out_path=os.path.join(output_folder, str(f'{ind:05}') + ".JPEG")
            print(f'{ind}. save to: ', out_path)
            gray_crop=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(out_path), gray_crop, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 100])

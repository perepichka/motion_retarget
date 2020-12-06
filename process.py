import re
import os
import glob
import argparse
import numpy as np
import json

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

parser = argparse.ArgumentParser(description='Process json to npy')
parser.add_argument('path', type=str, help='path to json files')
parser.add_argument('--padding_start', type=str, help='percentage of frames to discard at start', default=0.1)
parser.add_argument('--padding_end', type=str, help='percentage of frames to discard at end', default=0.05)
parser.add_argument('--threshold', type=float, help='confidence threshold', default=0.5)
args = parser.parse_args()


threshold = args.threshold

def to_npy(path)

    rl_path = os.path.realpath(path)

    json_files = glob.glob(os.path.join(rl_path, '*.json'))

    assert len(json_files) != 0, "No valid json files!"

    json_files.sort(key=natural_keys)

    nframes = len(json_files)
    njoints = 15
    naxes = 2

    frames = []

    print('parsing {} files/frames'.format(nframes))

    counter = 1

    for frame, json_file in enumerate(json_files):


        num_frames = 0

        with open(json_file, 'r') as f:
            loaded = json.load(f)
            people = loaded['people']
            print('{} people in file'.format(len(people)))

            valid_keypoints = []

            for person in people:
                data = np.array(person['pose_keypoints_2d']).reshape(25, 3)
                keypoints = data[:15,[0,1]]
                confidence = data[:15, 2]
                print(confidence.mean())
                if confidence.mean() > threshold:
                    valid_keypoints.append(keypoints)

            if len(valid_keypoints) != 1 or frame == len(json_files)-1:
                if len(frames) != 0:
                    if len(frames) >= 60:
                        write_frames = np.array(frames).swapaxes(0,-1)
                        np.save(os.path.join(args.path, '{}.npy'.format(counter)), write_frames)
                        with open(os.path.join(args.path, '{}.npy'.format(counter)), 'w+') as fr_file:
                            fr_file.write('{},{}'.format(frame-len(frames), frame))
                        counter += 1
                    frames = []
            else:
                frames.append(valid_keypoints[0])



motions = glob.glob(os.path.join(os.path.realpath(args.path), '*'))

for motion in motions:
    characters = glob.glob(os.path.join(os.path.realpath(motion), '*'))
    for character in characters:
        pth = os.path.realpath(path)
        to_npy(pth)


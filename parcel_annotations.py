import os
import json
from collections import defaultdict
import random

rootdir = './Datasets'
all = defaultdict(list)
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            ep = {}
            with open(os.path.join(subdir, file), 'r') as f:
                lines = f.readlines()
                ep['series'] = os.path.join(subdir, file).split('/')[-2]
                ep['metadata'] = "".join(lines[:5])
                ep['text'] = "".join(lines[5:50])
            ep['file_name'] = file
            all[ep['series']].append(ep)


counts = defaultdict(int)

for series, episodes in all.items():
    print(series)
    iaa = int(len(episodes) / 20)
    split = int((len(episodes) - iaa) / 4)
    random.shuffle(episodes)

    batch_0 = episodes[:iaa]
    batch_1 = episodes[iaa:iaa + split]
    batch_2 = episodes[iaa + split: iaa + split*2]
    batch_3 = episodes[iaa + split * 2: iaa + split*3]
    batch_4 = episodes[iaa + split * 3:]

    batches = {0: batch_0, 1: batch_1, 2: batch_2, 3: batch_3, 4: batch_4}

    for number, batch in batches.items():
        print(f'batch {number} size: ', len(batch))
        counts[number] += len(batch)
        with open(f'batch_{number}.jsonl', 'a') as json_out:
            for ep in batch:
                ep['batch_id'] = number
                json_out.write(json.dumps(ep))

print('\n', counts)












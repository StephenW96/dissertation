import os, fnmatch
import pandas as pd
import re

#dir_path = '/group/corporapublic/cslu_fae/speech/'
#dir_path = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_22_lang/speech/'
# dir_path = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_/speech/'
dir_path = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_22_merged/speech/'

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    print(result)
    return result

results = find('*.wav', dir_path)
# print(len(results))

data_list = []

for i in results:
    #path = re.search('/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_22_lang/speech/([\w]+)(/\d+)/.+\.wav',i)
    #path = re.search('/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_fae/speech/([\w]+)(/\d+)/.+\.wav',i)
    path = re.search('/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_22_merged/speech/([\w]+)(/\d+)/.+\.wav',i)

    path = path.groups()[0]+path.groups()[1]+'/'

    # path = re.search('/group/corporapublic/cslu_fae/speech/([\w]+)/.+.wav',i)

    name = os.path.basename(i)
    print(name)


    label = name[6:8]

    # path = name[1:3]+'/'

    #label = name[:2]
    data_list.append((path, name, label))

df = pd.DataFrame(data_list, columns=['path', 'name', 'label'])

df.to_csv('/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/annotations/cslu_22_aug.csv', index=False)

print(df)
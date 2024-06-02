import glob
import os
import random
from os.path import join
dataset_fp="train"
target_fp="train_set"
import pickle
def dataset_spliter(dataset_fp, target_fp,random_seed=1):
    ## split dataset set into train set and eval set
    mod_names = [fn for fn in os.listdir(dataset_fp)]
    mod_names = [fp for fp in mod_names if os.path.isdir(os.path.join(dataset_fp, fp))]
    source_file_paths=[]
    target_file_paths=[]
    for mod_name in mod_names:
        files=glob.glob(join(dataset_fp, mod_name, "*/**.npz"), recursive=True)
        random.seed(random_seed)
        files_envl=random.sample(files, int(len(files) / 100))
        files.remove(files_envl)
        for file in files:
            if not os.path.exists(os.path.join(target_fp, "train", mod_name, file.split('/')[-2])):
                os.makedirs(os.path.join(target_fp, "train", mod_name, file.split('/')[-2]))
                os.makedirs(os.path.join(target_fp, "envl", mod_name, file.split('/')[-2]))

            os.link(os.path.join(dataset_fp, mod_name,file.split('/'+mod_name+'/')[-1]), os.path.join(target_fp,"train",  mod_name,file.split('/'+mod_name+'/')[-1]))
            source_file_paths.append(os.path.join(dataset_fp, mod_name,file))
            target_file_paths.append(os.path.join(target_fp, "train",mod_name,file))
        for file in files_envl:
            os.link(os.path.join(dataset_fp, mod_name, file.split('/'+mod_name+'/')[-1]), os.path.join(target_fp,"envl", mod_name,file.split('/'+mod_name+'/')[-1]))
            source_file_paths.append(os.path.join(dataset_fp, mod_name, file))
            target_file_paths.append( os.path.join(target_fp,"envl", mod_name,file.split('/'+mod_name+'/')[-1]))

    with open('sample.pkl', 'wb') as f:
        pickle.dump([source_file_paths,target_file_paths], f)
        ##save a path that can help other device
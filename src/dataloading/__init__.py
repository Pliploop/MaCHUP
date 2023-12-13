from .augmentations import *
from .datasets import *
import pandas as pd
from src.dataloading.finetuning_dataset_tasks import *


class DatasetRouter:

    def __init__(self, task, data_dir, annotations_path="", augmentations=None, transform=True, target_sample_rate=24000, target_length=20, n_augmentations=1, extension="wav", sanity_check_n=None, validation_split=0.1, test_split=0.1) -> None:
        self.task = task
        self.data_dir = data_dir
        self.annotations_path = annotations_path
        self.augmentations = augmentations
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.n_augmentations = n_augmentations
        self.extension = extension
        self.sanity_check_n = sanity_check_n
        self.validation_split = validation_split
        self.test_split = test_split
        
        
        if self.task == 'GTZAN':
            annotations = self.get_gtzan_annotations(annotations_path)
            class2id = None
        elif self.task == 'MTGTop50Tags':
            annotations, class2id = self.get_mtg_top50_annotations(annotations_path)
            
            
            
        train_annotations = annotations[annotations["split"] == "train"]
        val_annotations = annotations[annotations["split"] == "val"]
        test_annotations = annotations[annotations["split"] == "test"]
            
            
        train_dataset_class = eval(self.task + "TrainDataset")
        val_dataset_class = eval(self.task + "TrainDataset")
        test_dataset_class = eval(self.task + "TestDataset")
        
        self.train_dataset = train_dataset_class(data_dir=self.data_dir, annotations=train_annotations, augmentations=self.augmentations, transform=self.transform, target_sample_rate=self.target_sample_rate,
                                                 target_length=self.target_length, n_augmentations=self.n_augmentations, extension=self.extension, sanity_check_n=self.sanity_check_n, train = True, class2id = class2id)

        self.val_dataset = val_dataset_class(data_dir=self.data_dir, annotations=val_annotations, augmentations=None, transform=False, target_sample_rate=self.target_sample_rate,
                                             target_length=self.target_length, n_augmentations=self.n_augmentations, extension=self.extension, sanity_check_n=self.sanity_check_n, train = False, class2id = class2id)
        
        self.test_dataset = test_dataset_class(data_dir=self.data_dir, annotations=test_annotations, augmentations=None, transform=False, target_sample_rate=self.target_sample_rate,
                                             target_length=self.target_length, n_augmentations=self.n_augmentations, extension=self.extension, sanity_check_n=self.sanity_check_n, train = False, class2id = class2id)
        
        
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_test_dataset(self):
        return self.test_dataset    
    
        
    def get_gtzan_annotations(self, annotations_path):
        annotations = pd.read_csv(os.path.join(
            annotations_path, "data/gtzan_annotations.csv"))


        # do a random split of the data into train, val and test, put this into the dataframe as a column "split"
        annotations["split"] = np.random.choice(["train", "val", "test"], size=len(annotations), p=[
                                                1-self.validation_split-self.test_split, self.validation_split, self.test_split])
        
        
        # add a column to annotations with the class index
        annotations["class_idx"] = annotations["genre"].astype(
            'category').cat.codes
        
        return annotations
    
    
    def get_mtg_top50_annotations(self,annotations_path):
        
        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_top50tags-split.tsv"

        #create a dataframe with paths an tags
        annotations = []
        
        
        class2id = {}
        for split in ['train', 'validation', 'test']:
            data = open(path.replace("split.tsv",f"{split}.tsv"), "r").readlines()
            all_paths = [line.split('\t')[3] for line in data[1:]]
            all_tags = [line.split('\t')[5:] for line in data[1:]]
            # annotations = annotations.append(pd.DataFrame({"path":all_paths, "tags":all_tags, "split":split}))
            # do this without append
            
            annotations.append(pd.DataFrame({"path":all_paths, "tags":all_tags, "split":split}))
            
            
            
            
            
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
                       
        annotations = pd.concat(annotations) 
        #replace mp3 extensions with wav in path columns
        if self.extension == "wav":
            annotations["path"] = annotations["path"].str.replace(".mp3", ".wav")
        print(annotations.head())

        return annotations, class2id
        
        
        


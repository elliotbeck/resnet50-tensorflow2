import h5py
import json

import tensorflow as tf
from tensorflow_datasets.core import BuilderConfig, dataset_utils
import tensorflow_datasets.public_api as tfds
from PIL import Image

import local_settings
import os
import numpy as np
from absl import flags, app

from random import shuffle

_CITATION = ""

_DESCRIPTION = """\
PCAS
"""

# see https://www.tensorflow.org/datasets/add_dataset

flags.DEFINE_list(name="validation_split", default=["photo"], help="")
flags.DEFINE_string(name="tfds_path", default=None, help="")

flags = flags.FLAGS

CELL_TYPES = {
    "HUVEC": 0,
    "RPE": 1,
    "HEPG2": 2,
    "U2OS": 3
}

VALIDATION_SPLIT = ["photo"]


class PACSConfig(tfds.core.BuilderConfig):
    def __init__(self, validation_split=None, **kwargs):
        self.validation_split = VALIDATION_SPLIT

        if validation_split is not None:
            self.validation_split = validation_split

        super(PACSConfig, self).__init__(
            name="{}".format("_".join(self.validation_split)),
            description="pacs dataset",
            version="0.2.0",
            **kwargs)


class PACS(tfds.core.GeneratorBasedBuilder):
    def __init__(self, validation_split=None, **kwargs):
        config = PACSConfig(validation_split=validation_split)
        super().__init__(config=config, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(
                    shape=(227, 227, 3), #TODO: add as argument and resize accordingly
                    encoding_format="png"),
                "attributes": {
                    "label": tf.int64,
                    "domain": tf.string
                }
            }),
            urls=["http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017"],
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        
        # TODO: remove split defined in validation_split and create separate test set for it
        # TODO: download remaining datasets and fix the filename vars below - done

        filenames = ['pacs/art_painting_train.hdf5', 'pacs/cartoon_train.hdf5',
                     'pacs/photo_train.hdf5']
        train_files = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = ['pacs/art_painting_val.hdf5', 'pacs/cartoon_val.hdf5',
                     'pacs/photo_val.hdf5']
        val_files_in = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = ['pacs/sketch_val.hdf5']
        val_files_out = [os.path.join(local_settings.RAW_DATA_PATH, f)
                       for f in filenames]

        filenames = ['pacs/art_painting_test.hdf5', 'pacs/cartoon_test.hdf5',
                     'pacs/photo_test.hdf5']
        test_files_in = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = ['pacs/sketch_test.hdf5']
        test_files_out = [os.path.join(local_settings.RAW_DATA_PATH, f)
                       for f in filenames]




        return [tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=30,
                    gen_kwargs=dict(
                        split="train",
                        files=train_files
                    )),

                tfds.core.SplitGenerator(
                    name=tfds.Split.VALIDATION,
                    num_shards=10,
                    gen_kwargs=dict(
                        split="validation",
                        files=val_files_in
                    )),

                tfds.core.SplitGenerator(
                    name="validation_out",
                    num_shards=10,
                    gen_kwargs=dict(
                        split="validation_out",
                        files=val_files_out
                )),
                    tfds.core.SplitGenerator(
                    name="test_in",
                    num_shards=1,
                    gen_kwargs=dict(
                        split="test_in",
                        files=test_files_in
                )),
                tfds.core.SplitGenerator(
                    name="test_out",
                    num_shards=1,
                    gen_kwargs=dict(
                        split="test_out",
                        files=test_files_out
                ))
                ]
    

    def _generate_examples(self, split, files):
        
        for f in files:
            file_ = h5py.File(f, 'r')
            images = list(file_['images'])
            labels = list(file_['labels'])

            for img, label_ in zip(images, labels):

                example = {
                    "attributes": {
                        "label": label_,
                        "domain": f.split("_")[0].split("/")[-1]
                    },
                    "image": np.uint8(img)
                }

                yield example




def main(_):
    builder_kwargs = {
        "validation_split": flags.validation_split
    }

    tfdataset_path = local_settings.TF_DATASET_PATH
    if flags.tfds_path is not None:
        tfdataset_path = flags.tfds_path

    train, dsinfo = tfds.load("pacs", 
        data_dir=tfdataset_path, split=tfds.Split.VALIDATION,
        builder_kwargs=builder_kwargs, with_info=True)

    for example in dataset_utils.as_numpy(train):
        import pdb; pdb.set_trace()
        print(example["attributes"]["label"])

if __name__ == '__main__':
    app.run(main)
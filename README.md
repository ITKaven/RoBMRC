# RoBMRC

Code and data of the paper "A Robustly Optimized BMRC for Aspect Sentiment Triplet Extraction, NAACL 2022" 

Authors: 	Shu Liu, Kaiwen Li , Zuhe Li

#### Requirements:

```
  python==3.8.5
  torch==1.9.0+cu111
  transformers==4.8.2
```

#### Original Datasets:

You can download the 14-Res, 14-Lap, 15-Res, 16-Res datasets from https://github.com/xuuuluuu/SemEval-Triplet-data.
Put it into different directories (./data/original/[v1, v2]) according to the version of the dataset.

#### Data Preprocess:

```
  python ./tools/DataProcessV1.py # Preprocess data from version 1 dataset
  python ./tools/DataProcessV2.py # Preprocess data from version 2 dataset
```
The results of data preprocessing will be placed in the ./data/preprocess/.

#### How to run:

```
  python ./tools/Main.py --mode train # For training
  python ./tools/Main.py --mode test # For testing
```
Training different versions of datasets can modify the value of dataset_version in Main.py.
```
dataset_version = "v1/"
dataset_version = "v2/"
```
#### Citation:
If you used the datasets or code, please cite our paper.
```
@inproceedings{liu-etal-2022-robustly,
    title = "A Robustly Optimized {BMRC} for Aspect Sentiment Triplet Extraction",
    author = "Liu, Shu  and
      Li, Kaiwen  and
      Li, Zuhe",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.20",
    doi = "10.18653/v1/2022.naacl-main.20",
    pages = "272--278",
}
```
#### Reference:
```
Shu Liu, Kaiwen Li, and Zuhe Li. 2022. A Robustly Optimized BMRC for Aspect Sentiment Triplet Extraction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 272–278, Seattle, United States. Association for Computational Linguistics.
```

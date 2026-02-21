# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""K-MHaS Korean Multi-label Hate Speech Dataset"""


import csv

import datasets


_CITATION = """\
@inproceedings{lee-etal-2022-k,
    title = "K-{MH}a{S}: A Multi-label Hate Speech Detection Dataset in {K}orean Online News Comment",
    author = "Lee, Jean  and
      Lim, Taejun  and
      Lee, Heejun  and
      Jo, Bogeun  and
      Kim, Yangsok  and
      Yoon, Heegeun  and
      Han, Soyeon Caren",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.311",
    pages = "3530--3538",
    abstract = "Online hate speech detection has become an important issue due to the growth of online content, but resources in languages other than English are extremely limited. We introduce K-MHaS, a new multi-label dataset for hate speech detection that effectively handles Korean language patterns. The dataset consists of 109k utterances from news comments and provides a multi-label classification using 1 to 4 labels, and handles subjectivity and intersectionality. We evaluate strong baselines on K-MHaS. KR-BERT with a sub-character tokenizer outperforms others, recognizing decomposed characters in each hate speech class.",
}
"""

_DESCRIPTION = """\
The K-MHaS (Korean Multi-label Hate Speech) dataset contains 109k utterances from Korean online news comments labeled with 8 fine-grained hate speech classes or Not Hate Speech class.
The fine-grained hate speech classes are politics, origin, physical, age, gender, religion, race, and profanity and these categories are selected in order to reflect the social and historical context.
"""

_HOMEPAGE = "https://github.com/adlnlp/K-MHaS"

_LICENSE = "cc-by-sa-4.0"

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt"
_VALIDATION_DOWNLOAD_URL = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_valid.txt"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_test.txt"

_CLASS_NAMES = [
    "origin",
    "physical",
    "politics",
    "profanity",
    "age",
    "gender",
    "race",
    "religion",
    "not_hate_speech"    
]

class Kmhas(datasets.GeneratorBasedBuilder):
    """K-MHaS Korean Multi-label Hate Speech Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.Sequence(datasets.ClassLabel(names=_CLASS_NAMES))
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate K-MHaS Korean Multi-label Hate Speech examples"""
        
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]
            
            for index, line in enumerate(lines):
                row = line.strip().split('\t')
                sentence = row[0]
                label = [int(ind) for ind in row[1].split(",")]
                yield index, {
                    "text" : sentence,
                    "label": label,              
                }


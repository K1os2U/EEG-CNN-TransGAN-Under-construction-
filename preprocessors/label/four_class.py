import numpy as np

from ..constant import LABEL_DICT
from ..base import _MicroPreprocessor


def genearte_binary_label(labels, trail_sample_num=60):
    assert 'Valence' in LABEL_DICT.keys()
    assert 'Arousal' in LABEL_DICT.keys()

    positive_valence_labels = labels[:, LABEL_DICT['Valence']] > 5
    positive_arousal_labels = labels[:, LABEL_DICT['Arousal']] > 5
    positive_labels = positive_valence_labels + 2 * positive_arousal_labels
    final_positive_labels = np.empty([0])
    for i in range(len(positive_labels)):
        for _ in range(0, trail_sample_num):
            final_positive_labels = np.append(final_positive_labels,
                                              positive_labels[i])
    return final_positive_labels


class BinaryLabel(_MicroPreprocessor):
    def __init__(self,
                 trail_sample_num=60,
                 start_trail_num=0,
                 end_trail_num=40):
        super().__init__()
        self.trail_sample_num = trail_sample_num
        self.start_trail_num = start_trail_num
        self.end_trail_num = end_trail_num

    def run(self, labels):
        labels = labels[self.start_trail_num:self.end_trail_num]
        return genearte_binary_label(labels, trail_sample_num=self.trail_sample_num)
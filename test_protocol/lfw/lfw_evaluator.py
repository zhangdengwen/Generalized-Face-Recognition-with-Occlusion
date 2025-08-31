"""
@author: Haoran Jiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import numpy as np
    
class LFWEvaluator(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        pair_list(list): the pair list given by PairsParser.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, data_loader, pairs_parser_factory, feature_extractor):
        """Init LFWEvaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            pair_list(list): the pair list given by PairsParser.
            feature_extractor(object): a feature extractor.
        """
        self.data_loader = data_loader
        pairs_parser = pairs_parser_factory.get_parser()
        self.pair_list = pairs_parser.parse_pairs()
        self.feature_extractor = feature_extractor

    def test(self, model):
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        mean, std = self.test_one_model(self.pair_list, image_name2feature)
        return mean, std

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize = True):
        subsets_score_list = np.zeros((10, 600), dtype = np.float32)
        subsets_label_list = np.zeros((10, 600), dtype = np.int8)
        valid_counts = np.zeros(10, dtype=int)  # 统计每子集有效对数

        for index, cur_pair in enumerate(test_pair_list):
            cur_subset = index // 600
            cur_id = index % 600
            image_name1, image_name2, label = cur_pair
            if(image_name1 not in image_name2feature or  image_name2 not in image_name2feature):
                continue
            subsets_label_list[cur_subset][cur_id] = label

            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]
            if not is_normalize:
                feat1 = feat1 / np.linalg.norm(feat1)
                feat2 = feat2 / np.linalg.norm(feat2)
            subsets_score_list[cur_subset][cur_id] = np.dot(feat1, feat2)
            valid_counts[cur_subset] += 1

        subset_train = np.array([True] * 10)
        accu_list = []
        for subset_idx in range(10):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True

            best_thres = self.getThreshold(train_score_list, train_label_list)

            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]

            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)

            valid_pair_num = np.sum(test_label_list == 1) + np.sum(test_label_list == 0)
            if valid_pair_num == 0:
                accu_list.append(0)
            else:
                accu_list.append((true_pos_pairs + true_neg_pairs) / valid_pair_num)

        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10)
        return mean, std

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size

        score_max = np.max(score_list) if score_list.size > 0 else 1
        score_min = np.min(score_list) if score_list.size > 0 else 0
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1)) 

        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums if neg_pair_nums > 0 else 0
            tpr = np.sum(pos_score_list > threshold) / pos_pair_nums if pos_pair_nums > 0 else 0
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr-fpr)
        best_thres = threshold_list[best_index]
        return  best_thres


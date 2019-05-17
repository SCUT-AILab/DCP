import torch
import random


class StratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, stratified_categories_index, num_samples_per_category):
        super(StratifiedSampler, self).__init__(object())
        self.stratified_categories_index = stratified_categories_index
        self.num_samples_per_category = num_samples_per_category

        self.lengths = [num_samples_per_category for _ in self.stratified_categories_index]
        self.size = sum(self.lengths)

    def __iter__(self):
        to_sample_list = []
        for length, stratified_category in zip(self.lengths, self.stratified_categories_index):
            to_sample_list += random.sample(stratified_category, length)
        random.shuffle(to_sample_list)
        return iter(to_sample_list)

    def __len__(self):
        return self.size


def get_stratified_categories_index(dataset):
    stratified_categories_index = [list() for _ in range(1000)]
    for index, (path, category) in enumerate(dataset.samples):
        stratified_categories_index[category].append(index)
    return stratified_categories_index

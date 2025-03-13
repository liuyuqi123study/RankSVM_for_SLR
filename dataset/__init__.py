from dataset.JsonFromFiles import JsonFromFilesDataset
from dataset.PairwiseDataset import PairwiseDataset
from dataset.TripletDataset import TripletDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "pairwise": PairwiseDataset,
    "triplet":TripletDataset
}

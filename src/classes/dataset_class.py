import torchvision.datasets as dset


path2data="dvc_data/train"
path2json="dvc_data/train/coco.json"

coco_train = dset.CocoDetection(root = path2data,
                                annFile = path2json)

print('Number of samples: ', len(coco_train))

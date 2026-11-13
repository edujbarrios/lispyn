; %% [markdown]
; ## ResNet18 transfer learning on labeled image folders
; Requires images in photos/class_name/*.jpg and a one-time weights download.
; %% [code]
(import torch)
(import torchvision)
(import lispyn.media [image-folder])
(import lispyn.training [trainable fit])

(setv weights torchvision.models.ResNet18_Weights.DEFAULT)
(setv model (torchvision.models.resnet18 :weights weights))
(setv photos (image-folder "photos" :transform (weights.transforms)))
(setv model.fc (torch.nn.Linear model.fc.in_features (len photos.classes)))
(trainable model "fc")
(setv loader (torch.utils.data.DataLoader photos :batch_size 16 :shuffle True))
(setv optimizer (torch.optim.Adam (.parameters model.fc) :lr 0.0001))
(print (fit model loader optimizer :loss-fn (torch.nn.CrossEntropyLoss) :epochs 3))

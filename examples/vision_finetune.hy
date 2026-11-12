; %% [markdown]
; ## Minimal image training, no downloads
; Switch FakeData to (image-folder "my_images") for real images.
; %% [code]
(import torch)
(import torchvision)
(import lispyn.training [fit trainable])
(import lispyn.data [make-dataloader])
(setv dataset (torchvision.datasets.FakeData :size 6 :image_size #(3 8 8)
                                         :num_classes 2 :transform (torchvision.transforms.ToTensor)))
(setv loader (make-dataloader dataset :batch-size 3))
(setv model (torch.nn.Sequential (torch.nn.Flatten) (torch.nn.Linear 192 2)))
(trainable model "1") ; train only classifier layer
(setv optimizer (torch.optim.SGD (filter (fn [p] p.requires_grad) (.parameters model)) :lr 0.01))
(setv history (fit model loader optimizer :loss-fn (torch.nn.CrossEntropyLoss) :epochs 1))
(print "train loss:" (get history "train_loss"))

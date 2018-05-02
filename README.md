PyTorch implementation of the ICLR 2018 paper [Learning to Pay Attention](https://openreview.net/forum?id=HyzbhfWRW).

Thanks for the baseline code [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

- VGG-ATT-PC:

  Tot: 100/100 | Loss: 0.182 | Acc: 95.260% (9526/10000)

- VGG-ATT-DP:

  Tot 100/100 | Loss: 0.196 | Acc: 94.900% (9490/10000)

- VGG-ATT (same CNN structure as VGG-ATT-PC and VGG-ATT-DP but no attention):

  Tot: 100/100 | Loss: 0.178 | Acc: 95.460% (9546/10000)

- VGG: 93.xxx%

(Seems main contribution is the higher resolution of early layers, not the attentions? :( Need more experiments on larger datasets like cifar-100 or ImageNet)



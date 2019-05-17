import dcp.models as md
import torch


def get_model(dataset, net_type, depth, n_classes):
    """
    Available model
    cifar:
        preresnet
        vgg
    imagenet:
        resnet
    """

    if dataset in ["cifar10", "cifar100"]:
        test_input = torch.randn(1, 3, 32, 32).cuda()
        if net_type == "preresnet":
            model = md.PreResNet(depth=depth, num_classes=n_classes)
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)

    elif dataset in ["imagenet", "sub_imagenet"]:
        test_input = torch.randn(1, 3, 224, 224).cuda()
        if net_type == "resnet":
            model = md.ResNet(depth=depth, num_classes=n_classes)
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)

    else:
        assert False, "unsupported data set: {}".format(dataset)
    return model, test_input

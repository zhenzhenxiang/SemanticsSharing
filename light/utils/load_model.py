
import os
import torch


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
        module state_dict inplace
        :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def check_modules(pretrained_dict, model_dict):
    num_dict = 0
    for k, v in pretrained_dict.items():
        num_dict += 1
        if (k in model_dict and v.size() == model_dict[k].size()):
            # print(num_dict, k, v.size())
            pass
        elif k not in model_dict:
            print(num_dict, "!!name error: ", k, v.size())
        elif v.size() != model_dict[k].size():
            print(num_dict, "!!size error: ", k, v.size(), model_dict[k].size())
        else:
            print(num_dict, "!!: ", k, v.size())


def load_model(resume,model):
    if resume:
        if os.path.isfile(resume):
            name, ext = os.path.splitext(resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            print('Resuming training, loading {}...'.format(resume))
            pretrained_dict = convert_state_dict(torch.load(resume, map_location=lambda storage, loc: storage))
            model_dict = model.state_dict()
            check_modules(pretrained_dict, model_dict)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                              (k in model_dict and v.size() == model_dict[k].size())}
            # self.pretrained_dict_mobilenetv3 = pretrained_dict_mobilenetv3
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print('checkpoints {} does not exist!'.format(resume))

    else:
        print('checkpoints {} does not apply!'.format(resume))

    return model

def load_modules(args, model):
    model = load_model(args.mobilenetv3_model_path, model)
    model = load_model(args.pwcnet_model_path, model)
    model = load_model(args.FFM60_model_path, model)
    model = load_model(args.FFM120_model_path, model)
    return model




def fix_model(args,model):
    # fix pwcnet
    if args.fixed_pwcnet:
        for v in model.flow_Network.parameters():
            v.requires_grad = False
    # fix mobilenetv3
    if args.fixed_mobilenetv3:
        for v in model.pretrained.parameters():
            v.requires_grad = False
        for v in model.head.parameters():
            v.requires_grad = False
    # fix FFM60
    if args.fixed_FFM60:
        for v in model.FFM_60.parameters():
            v.requires_grad = False
    # fix FFM120
    if args.fixed_FFM120:
        for v in model.FFM_120.parameters():
            v.requires_grad = False
    # print layers to train
    num_dict = 0
    for v in model.parameters():
        if v.requires_grad == True:
            num_dict += 1
    print('fixed layers number: ', num_dict)
    return model
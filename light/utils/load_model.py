
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
def load_model(resume,model):
    if resume:
        if os.path.isfile(resume):
            name, ext = os.path.splitext(resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            print('Resuming training, loading {}...'.format(resume))
            pretrained_dict_mobilenetv3 = convert_state_dict(torch.load(resume, map_location=lambda storage, loc: storage))
            model_dict = model.state_dict()
            pretrained_dict_mobilenetv3 = {k: v for k, v in pretrained_dict_mobilenetv3.items() if
                                           (k in model_dict and v.size() == model_dict[k].size())}
            # self.pretrained_dict_mobilenetv3 = pretrained_dict_mobilenetv3
            model_dict.update(pretrained_dict_mobilenetv3)
            model.load_state_dict(model_dict)
            return model
        else:
            print('mobilenetv3 ---->>>  checkpoints {} does not exist!'.format(resume))

    else:
        print('resume ---->>>  checkpoints {} does not apply!'.format(resume))
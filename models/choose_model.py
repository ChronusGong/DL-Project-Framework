from models.ResNet import BasicBlock, ResNet18, ResNet50
from models.DRSN import RSBU_CW
from models.EfficientNet import EfficientNetB0
from models.CNN import create_model
from models.transformer import TransformerPredictor


def choose_model(model_name):
    if model_name == 'ResNet34':
        model = ResNet18(BasicBlock, [3, 4, 6, 3], num_classes=1)
    elif model_name == 'ResNet18':
        model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=1)
    elif model_name == 'ResNet50':
        model = ResNet50(BasicBlock, [3, 4, 6, 3], num_classes=1)
    elif model_name == 'DRSN':
        model = ResNet18(RSBU_CW, [3, 4, 6, 4], num_classes=1)
    elif model_name == 'EfficientNet':
        model = EfficientNetB0(768, 1, )
    elif model_name == 'CNN':
        model = create_model()
    elif model_name == 'transformer':
        model = TransformerPredictor(input_size=768, hidden_size=1024, output_size=1)
    else:
        raise ValueError("illegal args")

    return model

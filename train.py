import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from untils import load_mnist
from untils import split_data
from untils import get_hidden_sizes

def define_argparser ():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config
def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d'%config.gpu_id)

    x,y = load_mnist(is_train=True, flatten = True)
    '''
    is_train = True는 학습 중 False라면 학습 아니고 모델 성능 테스트
    flatten = True는 다차원 텐서를 1차원으로 펴주는 거 True
    '''
    x,y = split_data(x.to(device), y.to(device), train_ratio = config.train_ratio)#x[[학습용 이미지 데이터],[검즘용 이미지 데이터]]로 만들어줌 y도 똑같이 수행 이떄 비율이 train_ratio = 8:2

    print("train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    torch.save({
        'model' : trainer.model.state_dict(),
        'opt' : optimizer.state_dict(),
        'config' : config
    }, config.model_fn)
if __name__ == '__main__':
    config = define_argparser()
    main(config)
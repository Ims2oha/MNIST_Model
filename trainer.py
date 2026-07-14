from copy import deepcopy
import numpy as np
import torch

class Trainer():

    def __init__(self,model,optimizer,crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        super().__init__()

    def _batchify(self,x,y,batch_size,random_split = True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device) # x.size(0)은 데이터의 총 갯수 torch.randperm( . . )은 0부터 총갯수(-1)까지 무작위로 배열해줌 device는 GPU에서 돌릴지 CPU에서 돌릴지 정해주는 거
            x = torch.index_select(x, dim=0, index=indices)#위에서 만든 indices 순서대로 데이터를 배치함 x와 y둘다 같은 순서로 하기에 레이블은 유지됨
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)#무작위 데이터들을 batch_size 만큼 자름
        y = y.split(batch_size, dim=0)

        return x,y
    
    def _train(self,x,y,config):
        self.model.train()

        x,y = self._batchify(x,y,config.batch_size)#config안에 하이퍼 파라미터들을 담아둠 그 안에서 batch_size 하이퍼 파라미터를 가져와서 사용

        total_loss = 0

        for i,(x_i, y_i) in enumerate(zip(x,y)):
            '''
            zip(x,y)는 쪼개진 입력 배치들의 묶음 x와 정답 배치 y를 똑같은 순서대로 짝을 지어줌
            enumerate는 지금이 몇번째 배치인지 번호(인덱스 i)를 매겨줌
            '''
            y_hat_i = self.model(x_i)#model을 통해 y'을 구함
            loss_i = self.crit(y_hat_i,y_i.squeeze())#squeeze는 데이터가 [[1],[2],[3]]이렇게 있을 때 [1,2,3]이렇게 차원을 없애줌

            self.optimizer.zero_grad()

            loss_i.backward()

            self.optimizer.step()
            
            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

            total_loss += float(loss_i)
 
        return total_loss / len(x)
    def _validate(self,x,y,config):
        self.model.eval()
        with torch.no_grad():
            x, y =self._batchify(x,y,config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x,y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i,y_i.squeeze())

                if config.verbose >= 2:
                    print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

                total_loss += float(loss_i)
            return total_loss / len(x)
    
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)
        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss
            best_model = deepcopy(self.model.state_dict())
        print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e" % (
            epoch_index + 1,
            config.n_epochs,
            train_loss,
            valid_loss,
            lowest_loss,
        ))

        self.model.load_state_dict(best_model)
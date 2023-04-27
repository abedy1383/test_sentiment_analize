#test 1
from os import listdir
from sys import stdout
from typing import Any
from random import seed
from torch.optim import Adam
from torchtext import datasets 
from pydantic import BaseModel 
from functools import lru_cache
from torch.backends import cudnn
from torchtext.data import LabelField , Field , BucketIterator
from torch.nn import Module , Embedding , Linear , BCEWithLogitsLoss , LSTM , Dropout , init
from torch import manual_seed , cuda , device , no_grad , Tensor , sigmoid , round , load , save , cat , float as torch_float 

manual_seed(1383)
cuda.manual_seed(1383)
cudnn.deterministic = True

class DataClass:
    def __init__(self) -> None:
        self._device = device("cuda" if cuda.is_available() else "cpu")
    
    @property
    @lru_cache
    def device(self):
        return self._device
    
    @device.setter
    def seter(self, value : str = "cpu"):
        self._device = device(value)
    
    @device.deleter
    def deleter(self):
        del self._device
        
class BaseData(BaseModel):
    train : Any = None 
    test : Any = None 
    iter_next : Any = None 

class BaesOptimModel(BaseModel):
        optimizer : Any 
        loss_func : Any 
        model : Any 
        basedata : Any 

class Datasets():
    def __init__(self , 
            random_id : int = 999 , 
            batch_size : int = 64 , 
            _device : device = DataClass().device
        ) -> None:
        (   self._random_id , 
            self._batch_size , 
            self._device , 
            self._id_cash 
        ) = (
            random_id , 
            batch_size , 
            _device , 
            "1"
        )
        
        # run function 
        self._data()
        self._run(self._id_cash)
     
    def __len__(self) -> int:
        return len(self._text.vocab)
    
    def _data(self):
        self._text , self._label = Field(tokenizer_language="en_core_web_sm" , tokenize="spacy") , LabelField(dtype = torch_float)
   
    def _one_hot(self , _):
        # _ => self.test , self.train , self.valid
        self._text.build_vocab(_[1], max_size= len(_[1]) + len(_[2]), vectors="glove.6B.100d")
        self._label.build_vocab(_[1])
        
        return _
    
    @lru_cache
    def _run(self , _):
        return BaseData(
            test= (_itoreitor := BucketIterator.splits(
                datasets=self._one_hot(
                    (
                        # test
                        (_data := datasets.IMDB.splits(self._text , self._label))[1] , 
                        # train
                        (_train := _data[0].split(random_state=seed(self._random_id)))[0] , 
                        # valid
                        _train[1]
                    )
                ),
                batch_size=self._batch_size,
                device=self._device,
                shuffle=True
                )
            )[0],
            train=_itoreitor[1],
            iter_next = iter(_itoreitor[2])
        )
    
    def show(self):
        return self._run(self._id_cash )

class Model(Module):
    def __init__(self , 
            vocab_size ,
            embadding_dim , 
            hidden_dim , 
            output_dim ,
            dropout = 0.5,
            bidirectional = True
        ) -> None:
        super().__init__()
    
        self.emb = Embedding(vocab_size, embadding_dim , device=DataClass().device)
        self.rnn = LSTM(embadding_dim, hidden_dim, num_layers=2, 
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = Linear(hidden_dim*2, output_dim , device=DataClass().device)
        self.drp= Dropout(dropout)
    
    def forward(self , text):
        return self.fc(
            self.drp(
                cat(
                    (
                        (hidden:=self.rnn(
                            self.drp(
                                self.emb(
                                        text
                                    )
                                )
                            )[1][0]
                        )[-2,:,:] , 
                        hidden[-1,:,:]
                    ) , 
                    dim=1,
                )
            ).squeeze(0)
        )
         
class RunModel():
    def __init__(self , 
                baseinput : BaesOptimModel,
                _namept :str = "static_model.pt",
            ) -> None:
        ( 
            self._model , 
            self._optimizer , 
            self._loss_func , 
            self._basedata ,
            self._data ,
            self._loss ,
            self._namept ,
        ) = ( 
            baseinput.model , 
            baseinput.optimizer , 
            baseinput.loss_func , 
            baseinput.basedata , 
            None , 
            None , 
            _namept
        )
        
    def _analize(self ,
                epoch_j:int , 
                epoch_a:int , 
                train_j:int , 
                train_a:int , 
                loss:float = 0.0 , 
                best: Tensor = Tensor([0]),
                train:bool=True,
                test : float = 0.0
            )->None:
        stdout.write(f"""\r Repetitive round : [{epoch_j+1}/{epoch_a}]({int((epoch_j+1/epoch_a)*100)}%) / {("Training" if train else "Testing")} round : {train_j}/{train_a}]({int((train_j/train_a)*100)}%) / Loss : [{loss:.2f}] / The most diagnosis : [{best}] / Percent : [{test}%]""")
        stdout.flush()
    
    def _acurry(self , pread , y):
        return ((round(sigmoid(pread)) == y).float()).sum()
    
    def _check(self):
        if self._namept in listdir("./"):
            self._model.load_state_dict(load(f"./{self._namept}"))
    
    def train(self , epoch : int = 2 ):
        self._check()
        for _epoch in range(epoch):
            _list = None
            for _ , batch in enumerate(self._basedata.train):
                try:
                    # train model
                    self._model.train()
                    self._optimizer.zero_grad()
                    (loss := self._loss_func((_y_pread:=self._model(batch.text).squeeze(1)) , batch.label)).backward()
                    self._loss = (loss.item() , (self._loss[1] + 1 if self._loss is not None else 1))
                    self._optimizer.step()

                    _list = (
                        ((_list[0] if _list is not None else 0)+self._acurry(_y_pread,batch.label)) , _ + 1
                    )
   
                    if _ >= 10 or _epoch > 0:
                        # test model
                        self._model.eval()
                        self._data = ((self._model.state_dict() , _acurry_epoch) if (_acurry_epoch := self._acurry(self._model((_test := next(self._basedata.iter_next)).text).squeeze(1) , _test.label)) >= (self._data[1] if (_test_if := self._data is not None) else 0) else (self._data if _test_if else ("" , 0)))
                    
                    if _ % 10 == 0:
                        self._analize(
                        epoch_j=_epoch , 
                        epoch_a=epoch , 
                        train_a=len(self._basedata.train),
                        train_j=_,
                        loss=self._loss[0],
                        best=(self._data[1] if self._data is not None else 0),
                        test=_list[0]/_list[1]
                    )
                except:
                    pass 
        self._model.load_state_dict(self._data[0])
        save(self._model.state_dict() , "./state_model.pt")

    def test(self):
        self._model.eval()
        with no_grad():
            for _ , _batch in enumerate(self._basedata.test):
                try:
                    self._analize(
                        train=False,
                        epoch_a=1 , 
                        epoch_j= 0 , 
                        train_a= len(self._basedata.test) , 
                        train_j=_ , 
                        best=self._acurry(
                            self._model(_batch.text).squeeze(1),
                            _batch.label
                        )
                    )
                except:
                    pass 
                    
class OptemModel():
    def __init__(self,
        _model : Model,
        _optimizer : Adam,
        _basedata : Datasets,
        _loss_func : BCEWithLogitsLoss,
        _driver : device = DataClass().device,
        _block_liear : list = ["emb.weight"],
        _xavier_uniform : list = ["drp","fc","rnn"],
                   
    ) -> None:
        (
            self._model,
            self._optimizer,
            self._basedata,
            self._loss_func,
            self._driver,
            self._bloack ,
            self._xavier_uniform
        ) = (
            _model,
            _optimizer,
            _basedata,
            _loss_func,
            _driver,
            _block_liear,
            _xavier_uniform
        )
    
        # run function 
        self._device()
        self._UpdateAllWeights()
        self._OptemLiear() 

    def _device(self):
        self._model.to(self._driver)
        self._loss_func.to(self._driver)
        
    def init_weights(self , m):
        if isinstance(m, Linear):
            init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def _UpdateAllWeights(self):
        for _name , _tensor in self._model.named_parameters():
            if _tensor.dim() > 2:
                init.xavier_uniform(_tensor)
            
    def _OptemLiear(self):
        # copy weights "glove.6B.100d" in Model.weights
        self._model.emb.weight.data.copy_(self._basedata._text.vocab.vectors)
        
        self._model.apply(self.init_weights)

        for name , program in self._model.named_parameters():
            if name in self._bloack:
                program.requires_grad=False 
                
    def show(self) -> BaesOptimModel:
        return BaesOptimModel(
            optimizer=self._optimizer,
            loss_func=self._loss_func,
            model=self._model,
            basedata=self._basedata.show()
        )
              
_ = RunModel(
    baseinput=OptemModel(
        _basedata = (_vocab_size := Datasets()),
        _model = (_model := Model(len(_vocab_size) , 100 , 256 , 1)),
        _optimizer = Adam(_model.parameters() , lr= 1e-3),
        _loss_func = BCEWithLogitsLoss()
    ).show()
)
_.train(2)            
                
_.test()                
            
            
        

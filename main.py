import torch , sys 
from pydantic import BaseModel 
from typing import Any
import random 
from torch.nn import Module , Embedding , Linear , RNN , BCEWithLogitsLoss , LSTM , Dropout
from torchtext import datasets 
from torch.optim import Adam
from torchtext.data import LabelField , Field , BucketIterator
import os 

torch.manual_seed(1383)
torch.cuda.manual_seed(1383)
torch.backends.cudnn.deterministic = True

class BaseData(BaseModel):
    train : Any = None 
    test : Any = None 
    valid : Any = None 
    iter_next : Any = None 

class Datasets():
    def __init__(self , 
            random_id : int = 999 , 
            batch_size : int = 64 , 
            _device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        (   self._random_id , 
            self._batch_size , 
            self._device
        ) = (
            random_id , 
            batch_size , 
            _device
        )
        
        # run function 
        self._data()
        self._run()
        self._one_hot()
        self._itreitor()
     
    def __len__(self) -> int:
        return self._sum_data
    
    def _data(self):
        self._text , self._label = Field(tokenizer_language="en_core_web_sm" , tokenize="spacy") , LabelField(dtype = torch.float)
     
    def _run(self):
        self.train , self.test = datasets.IMDB.splits(self._text , self._label)
        self.train , self.valid = self.train.split(random_state=random.seed(self._random_id))
        self._sum_data = len(self.train) + len(self.valid)
        

    def _one_hot(self):
        self._text.build_vocab(self.train, max_size=self._sum_data)
        self._label.build_vocab(self.train)
        
    def _itreitor(self):
        self.train , self.valid , self.test = BucketIterator.splits(
            (self.train , self.valid , self.test),
            batch_size=self._batch_size,
            device=self._device,
            shuffle=True
        )
    
    def show(self):
        return BaseData(
            train= self.train, 
            test= self.test, 
            valid= self.valid,
            iter_next = iter(self.valid)
        )

class FakeLinear(Module):
    def __init__(self , hidden_dim , output_dim) -> None:
        super().__init__()
        self.fc = Linear(hidden_dim , output_dim)
    
    def forward(self , _ : torch.Tensor):
        return self.fc(_.squeeze(0))
    
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
    
        self.emb = Embedding(vocab_size, embadding_dim)
        self.rnn = LSTM(embadding_dim, hidden_dim, num_layers=2, 
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = FakeLinear(hidden_dim*2, output_dim)
        self.drp= Dropout(dropout)
    
    def forward(self , text):
        return self.fc(
            self.drp(
                torch.cat(
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
                    dim=1
                )
            )
        )
         
class RUNMODEL():
    def __init__(self , 
                model : Model, 
                basedata : BaseData, 
                loss_func : BCEWithLogitsLoss, 
                optimizer :Adam,
                _driver = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                _namept :str = "static_model.pt",
            ) -> None:
        ( 
            self._model , 
            self._optimizer , 
            self._loss_func , 
            self._basedata ,
            self._driver , 
            self._data ,
            self._loss ,
            self._namept ,
        ) = ( 
            model , 
            optimizer , 
            loss_func , 
            basedata , 
            _driver ,
            None , 
            None , 
            _namept
        )
        
        # run function 
        self._device()
         
    def _device(self):
        self._model.to(self._driver)
        self._loss_func.to(self._driver)
    
    def _analize(self ,
                epoch_j:int , 
                epoch_a:int , 
                train_j:int , 
                train_a:int , 
                loss:float = 0.0 , 
                best:torch.Tensor = torch.Tensor([0]),
                train:bool=True,
            )->None:
        sys.stdout.write(f"""\r Repetitive round : [{epoch_j+1}/{epoch_a}]({int((epoch_j+1/epoch_a)*100)}%) / {("Training" if train else "Testing")} round : {train_j}/{train_a}]({int((train_j/train_a)*100)}%) / Loss : [{loss}] / The most diagnosis : [{best}]""")
        sys.stdout.flush()
    
    def _acurry(self , pread , y):
        return ((torch.round(torch.sigmoid(pread)) == y).float()).sum()
    
    def _check(self):
        if self._namept in os.listdir("./"):
            self._model.load_state_dict(torch.load(f"./{self._namept}"))
    
    def train(self , epoch : int = 2 ):
        self._check()
        for _epoch in range(epoch):
            for _ , batch in enumerate(self._basedata.train):
                try:
                    # train model
                    self._model.train()
                    self._optimizer.zero_grad()
                    (loss := self._loss_func(self._model(batch.text).squeeze(1) , batch.label)).backward()
                    self._loss = (loss.item() , (self._loss[1] + 1 if self._loss is not None else 1))
                    self._optimizer.step()
                
                    if _ >= 10 or _epoch > 0:
                        # test model
                        self._model.eval()
                        with torch.no_grad():
                              self._data = ((self._model.state_dict() , _acurry_epoch) if (_acurry_epoch := self._acurry(self._model((_test := next(self._basedata.iter_next)).text).squeeze(1) , _test.label)) >= (self._data[1] if (_test_if := self._data is not None) else 0) else (self._data if _test_if else ("" , 0)))
                    if _ % 10 == 0:
                        self._analize(
                        epoch_j=_epoch , 
                        epoch_a=epoch , 
                        train_a=len(self._basedata.train),
                        train_j=_,
                        loss=self._loss[0],
                        best=(self._data[1] if self._data is not None else 0)
                    )
                except:
                    pass 
        self._model.load_state_dict(self._data[0])
        torch.save(self._model.state_dict() , "./state_model.pt")

    def test(self):
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
                    
_ = RUNMODEL(
    basedata= (_vocab_size := Datasets(_device = torch.device("cpu"))).show(),
    model= (_model := Model(len(_vocab_size) , 100 , 256 , 1)), 
    loss_func=BCEWithLogitsLoss(),
    optimizer=Adam(_model.parameters() , lr= 1e-3),
    _driver = torch.device("cpu")
)                 

_.train(2)            
                
_.test()                
            
            
        

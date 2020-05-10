from torch import tensor
import torch.optim as optim
from .callback import *
from .scheduler import *

class DataBunch:
    '''
    container class of train&valid data
    '''
    def __init__(self, train_dl, valid_dl):
        self.train_dl, self.valid_dl = train_dl, valid_dl
        self.train_ds, self.valid_ds = self.train_dl.dataset, self.valid_dl.dataset if self.valid_dl is not None else None
        self.valid_size = len(self.valid_ds)/(len(self.valid_ds)+len(self.train_ds)) if self.valid_dl is not None else 0.
        
class Learner:
    '''
    Container class of model, loss, optimizer, and data
    '''
    def __init__(self, model, data, loss_func, optimizer, cbs=None, cbfs=[partial(AvgStatsCallback,None)]):
        self.model,self.data,self.loss_func = model,data,loss_func
        self.opt = optimizer
        self.in_train,self.logger = False,print
        
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cbfs))

    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)
            
    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)
            
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            # yb is packed in a collection here and cannot be moved to device at this point
            self.xb,self.yb = xb,yb;                        self('begin_batch')
            self.yb = [y.to(self.model.device) for y in self.yb]
            self.pred = self.model(self.xb);                self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb); self('after_loss')
            if not self.in_train: return
            self.loss.backward();                           self('after_backward')
            self.opt.step();                                self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                        self('after_cancel_batch')
        finally:                                            self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i,(xb,yb) in enumerate(self.dl): self.one_batch(i, xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def do_begin_fit(self, epochs):
        self.epochs,self.loss = epochs,tensor(0.)
        self('begin_fit')

    def do_begin_epoch(self, epoch):
        self.epoch,self.dl = epoch,self.data.train_dl
        return self('begin_epoch')
    
    def fit(self, epochs, cbs=None):
        # pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        
        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                if not self('begin_epoch'): self.all_batches()
                
                if self.data.valid_size>0:
                    with torch.no_grad():
                        self.dl = self.data.valid_dl
                        if not self('begin_validate'): self.all_batches()
                
                self('after_epoch')
            
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.remove_cbs(cbs)
            
    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
    'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
    'begin_epoch', 'begin_validate', 'after_epoch',
    'after_cancel_train', 'after_fit'}
    
    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res
    
    def find_lr(self,min_lr=1e-4, max_iter=500):
        
        def get_required_fit_round(max_iter):
            round_up_div = lambda x,y: x//y + (1 if x%y > 0 else 0)
            return round_up_div(max_iter,len(self.data.train_dl))

        self.fit(get_required_fit_round(max_iter), [LR_Find(max_iter=max_iter,min_lr=min_lr), Recorder()])
    
    def plot_lr(self):
        self.recorder.plot()

    def fit1cycle(self,epochs,lr):

        sched_lr = combine_scheds([0.3, 0.7], [sched_cos(lr/5,lr),sched_cos(lr,lr/10)])
        sched_beta1 = combine_scheds([0.3, 0.7], [sched_cos(0.9, 0.8), sched_cos(0.8, 0.9)])
        sched_beta2 = combine_scheds([0.3, 0.7], [sched_cos(0.99, 0.9), sched_cos(0.9, 0.99)])
        def sched_betas(pos): return [sched_beta1(pos), sched_beta2(pos)]

        self.fit(epochs, [Recorder(),ParamScheduler('lr',sched_lr),ParamScheduler('betas',sched_betas)])

    def fit_with_sched(self, epochs, sched_dict):
        cbs = [Recorder()] + [ParamScheduler(param, sched_func) for param, sched_func in sched_dict.items()]
        self.fit(epochs, cbs)
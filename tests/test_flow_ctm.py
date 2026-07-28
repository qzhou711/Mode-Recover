import copy
import torch
from torch import nn
from agents.models.flow_matching.ctm import ctm_paths, freeze

class ToyFlow(nn.Module):
    def __init__(self,value):
        super().__init__(); self.value=nn.Parameter(torch.tensor(float(value)))
    def velocity(self,x,time,state): return torch.ones_like(x)*self.value
    def transition(self,x,start,stop,state):
        shape=(start.shape[0],)+(1,)*(x.ndim-1)
        return x+(stop-start).reshape(shape)*self.velocity(x,start,state)
    def boundary_transition(self,x,start,stop,state):
        shape=(start.shape[0],)+(1,)*(x.ndim-1)
        endpoint=x+(1-start).reshape(shape)*self.velocity(x,start,state)
        ratio=((1-stop)/(1-start).clamp_min(1e-6)).reshape(shape)
        return ratio*x+(1-ratio)*endpoint


def test_boundary_conditions_are_exact():
    model=ToyFlow(2.0)
    x=torch.randn(4,5,2); state=torch.randn_like(x)
    t=torch.tensor([0.0,0.2,0.5,0.9])
    torch.testing.assert_close(model.boundary_transition(x,t,t,state),x)
    one=torch.ones_like(t); shape=(4,1,1)
    expected=x+(1-t).reshape(shape)*model.velocity(x,t,state)
    torch.testing.assert_close(model.boundary_transition(x,t,one,state),expected)

def test_exact_constant_flow_has_zero_ctm_residual():
    student=ToyFlow(2.0); target=freeze(copy.deepcopy(student)); teacher=freeze(copy.deepcopy(student))
    action=torch.randn(4,5,2); state=torch.randn(4,5,2); noise=torch.randn_like(action)
    pred,ref,_,_=ctm_paths(student,target,teacher,action,state,noise,(3,11),16)
    torch.testing.assert_close(pred,ref)

def test_gradient_only_reaches_online_student():
    student=ToyFlow(1.5); target=freeze(ToyFlow(0.7)); teacher=freeze(ToyFlow(2.0))
    action=torch.randn(4,5,2); state=torch.randn(4,5,2); noise=torch.randn_like(action)
    pred,ref,_,_=ctm_paths(student,target,teacher,action,state,noise,(2,9),16)
    (pred-ref).square().mean().backward()
    assert student.value.grad is not None and student.value.grad.abs()>0
    assert target.value.grad is None and teacher.value.grad is None

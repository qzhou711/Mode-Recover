import torch


def pseudo_huber(prediction, target, delta=0.01):
    error=prediction-target
    return (torch.sqrt(error.square()+delta*delta)-delta).mean()


def freeze(module):
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    return module


@torch.no_grad()
def update_ema(target,online,decay):
    for tp,op in zip(target.parameters(),online.parameters()): tp.mul_(decay).add_(op,alpha=1.0-decay)


def conditional_distance_loss(student_endpoint, teacher_endpoint, samples_per_state):
    if samples_per_state <= 1:
        return student_endpoint.sum() * 0.0
    groups = student_endpoint.shape[0] // samples_per_state
    student = student_endpoint.flatten(1).reshape(groups, samples_per_state, -1)
    teacher = teacher_endpoint.flatten(1).reshape(groups, samples_per_state, -1)
    student_dist = torch.cdist(student, student)
    teacher_dist = torch.cdist(teacher, teacher)
    scale = teacher_dist.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return torch.nn.functional.smooth_l1_loss(student_dist / scale, teacher_dist / scale)


def ctm_paths(student,target,teacher,action,state,noise,indices,bins):
    t_index,s_index=indices; u_index=t_index+1
    batch,dtype,device=action.shape[0],action.dtype,action.device
    t=torch.full((batch,),t_index/bins,dtype=dtype,device=device)
    u=torch.full((batch,),u_index/bins,dtype=dtype,device=device)
    s=torch.full((batch,),s_index/bins,dtype=dtype,device=device)
    view=(batch,)+(1,)*(action.ndim-1)
    x_t=(1-t.reshape(view))*noise+t.reshape(view)*action
    with torch.no_grad():
        dt=(u-t).reshape(view); v_t=teacher.velocity(x_t,t,state)
        predictor=x_t+dt*v_t; v_u=teacher.velocity(predictor,u,state)
        x_u=x_t+0.5*dt*(v_t+v_u)
        target_s=target.boundary_transition(x_u,u,s,state)
    prediction_s=student.boundary_transition(x_t,t,s,state); one=torch.ones_like(s)
    if s_index==bins: prediction_final,target_final=prediction_s,target_s
    else:
        prediction_final=target.boundary_transition(prediction_s,s,one,state)
        with torch.no_grad(): target_final=target.boundary_transition(target_s,s,one,state)
    return prediction_final,target_final,x_t,t

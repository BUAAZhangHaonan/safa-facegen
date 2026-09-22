"""Differentiable Torch translation of SciPy's Dormand–Prince RK45 algorithm.

Source: scipy.integrate._ivp.{rk,common}, SciPy 1.17.0 (BSD-3-Clause).
The source license is retained in vendor/rectified_flow/SCIPY_LICENSE.txt.
Step selection is discrete; every accepted state update retains its input graph.
"""
import math
import torch

_C=(0.,1/5,3/10,4/5,8/9,1.)
_A=((),(1/5,),(3/40,9/40),(44/45,-56/15,32/9),
    (19372/6561,-25360/2187,64448/6561,-212/729),
    (9017/3168,-355/33,46732/5247,49/176,-5103/18656))
_B=(35/384,0.,500/1113,125/192,-2187/6784,11/84)
_E=(-71/57600,0.,71/16695,-71/1920,17253/339200,-22/525,1/40)


def _rms(value):
    result=float(torch.linalg.vector_norm(value.detach().reshape(-1))/math.sqrt(value.numel()))
    if not math.isfinite(result):raise FloatingPointError('Non-finite RF solver state or error estimate')
    return result


def _combine(values,weights):
    coefficients=values[0].new_tensor(weights)
    return torch.stack(values,dim=-1).matmul(coefficients)


def integrate(function,initial,t0=1e-3,t_bound=1.,rtol=1e-5,atol=1e-5,max_steps=10000):
    if initial.dtype!=torch.float64:raise TypeError('RK45 requires FP64 integration state')
    if not t0<t_bound or rtol<=0 or atol<=0:raise ValueError('Invalid RK45 interval or tolerance')
    y=initial;t=float(t0);interval=float(t_bound-t0)
    f=function(t,y)
    # SciPy select_initial_step: fourth-order embedded estimate and RMS scaling.
    with torch.no_grad():
        scale=atol+y.abs()*rtol
        d0,d1=_rms(y/scale),_rms(f/scale)
        h0=1e-6 if d0<1e-5 or d1<1e-5 else .01*d0/d1
        h0=min(h0,interval)
        f1=function(t+h0,y+h0*f)
        d2=_rms((f1-f)/scale)/h0
        h1=max(1e-6,h0*1e-3) if d1<=1e-15 and d2<=1e-15 else (.01/max(d1,d2))**.2
        h_abs=min(100*h0,h1,interval)
    attempts=0
    while t<t_bound:
        minimum=10*abs(math.nextafter(t,math.inf)-t)
        h_abs=max(h_abs,minimum)
        rejected=False
        while True:
            attempts+=1
            if attempts>max_steps:raise RuntimeError('RF RK45 exceeded its maximum step count')
            if h_abs<minimum:raise RuntimeError('RF RK45 step size underflow')
            t_new=min(t+h_abs,t_bound);h=t_new-t
            stages=[f]
            for coefficients,fraction in zip(_A[1:],_C[1:]):
                stages.append(function(t+fraction*h,y+h*_combine(stages,coefficients)))
            y_new=y+h*_combine(stages,_B)
            f_new=function(t_new,y_new)
            stages.append(f_new)
            with torch.no_grad():
                scale=atol+torch.maximum(y.abs(),y_new.abs())*rtol
                error_norm=_rms(h*_combine(stages,_E)/scale)
            if error_norm<1:
                factor=10. if error_norm==0 else min(10.,.9*error_norm**(-.2))
                if rejected:factor=min(1.,factor)
                h_abs=h*factor
                break
            h_abs=h*max(.2,.9*error_norm**(-.2));rejected=True
        t=t_new;y=y_new;f=f_new
    return y

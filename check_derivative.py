import autograd.numpy as np
from autograd import grad
import math
premium  = 3300
bench_prob = 0.5319



def fun_x(x):
    #effort = 10*(1-math.exp(-incentive/400))
    #renew_prob = 20*(1-math.exp(-effort/5
    y = 20*(1-math.exp(-(10*(1-math.exp(-(x)/400)))/5))
    return y 
def fct(x):
    y = x**2+1
    return y
def d_fun_x(x):
    return (math.exp(2*(math.exp(-1*x/400))-(x/400)-2))/10

grad_fct = grad(fct)
print(grad_fct(1.0))
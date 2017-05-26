'''
Kalman filter

Depending on initial values, it gave quite differenct results. Also, by ordering method affects the returns.

Return 187.2%
Alpha 0.11
Beta 0.46
Sharpe 1.48
Drawdown -11.3%
'''

import numpy as np
import pytz
import copy


def initialize(context):
    context.boeing = sid(698)
    context.lockeed = sid(12691)
   
    context.pos = None 
    context.day = None
    
    context.kf = copy.copy(context)
    
    ## Initial parameter
    context.kf.n = 10
    context.kf.Q = 1e-5
    context.kf.R = 0.1**2
    context.kf.K = np.zeros(context.kf.n) 
    context.kf.P = np.zeros(context.kf.n) 
    context.kf.Pminus = np.zeros(context.kf.n)  
    context.kf.x_hat = np.zeros(context.kf.n)      
    context.beta = np.zeros(context.kf.n)  
    context.kf.z = np.random.normal(1, 0.1, context.kf.n) 

def handle_data(context, data):
    exchange_time = get_datetime().astimezone(pytz.timezone('US/Eastern'))
    #Set order time
    if exchange_time.hour == 15 and exchange_time.minute >= 55:
        if context.day is not None and context.day == exchange_time.day:
            return
        context.day = exchange_time.day
        
        kf = context.kf
        
        boeing_hist = np.asarray(data.history(context.boeing, "price", kf.n, "1m")).reshape((1, kf.n))
        y = data.current(context.lockeed, 'price')
        yhat = boeing_hist.dot(context.beta)
        e = y - yhat
        
        kf.x_hat[0] = yhat
        kf.P[0] = 0.001

        for k in range(1, kf.n):
            kf.Pminus[k] = kf.P[k-1] + kf.Q
            kf.K[k] = kf.Pminus[k] / (kf.P[k] + kf.R)
            kf.x_hat[k] = kf.x_hat[k-1] + kf.K[k] * (kf.z[k] - kf.x_hat[k-1])
            kf.P[k] = (1 - kf.K[k]) * kf.Pminus[k]
        
        context.beta = context.beta + kf.K.flatten()
        sqrt_Q = np.sqrt(kf.Q)
        
        #Trade
        #close positions
        if context.pos is not None:
            if context.pos == 'long' and e > -sqrt_Q :
                order_target(context.boeing, 0)
                context.pos = None
            elif context.pos == 'short' and e < sqrt_Q :
                order_target(context.lockeed, 0)
                context.pos = None
        
        #open positions
        if context.pos is None:
            if e < -sqrt_Q :
                order(context.lockeed, 10000)
                boeing_amount = context.portfolio.positions[context.boeing].amount  
                order(context.boeing, -boeing_amount)
                context.pos = 'long'
            elif e > sqrt_Q :
                lockeed_amount = context.portfolio.positions[context.boeing].amount  
                order(context.lockeed, -lockeed_amount)
                order(context.boeing, 10000 * context.beta[0])
                context.pos = 'short'


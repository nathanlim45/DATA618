import datetime as dt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np
import math

'''
Retruns 89.5%
Alpha 0.08
Sharpe 0.80
Drawdown -15%
'''


max_trades = 2

def is_stationary(x, p):
    
    x = np.array(x)
    
    result = ts.adfuller(x, regression='ctt')
    # ts.adfuller returns adf, pvalue, usedlag, nobs, critical values...
    
    #if DFStat <= critical value
    if result[0] >= result[4][str(p) + '%']:
        #is stationary
        return True
    else:
        #is nonstationary
        return False
 
def coint_test(x, y):
    #check stationary with Augmented Dickey Fuller
    x_is_l1 = not(is_stationary(x, 10))
    y_is_l1 = not(is_stationary(y, 10))
    
    #if x and y are not stationary     
    
    if x_is_l1 and y_is_l1: 
        result = ts.coint(x, y)
        if result[0] >= result[2][2]:
            return True
        else:
            return False
    else:
        return False

def initialize(context):
    context.stocks = [
                     (sid(4283), sid(5885)),   # Coke, Pepsi
                     (sid(8229), sid(21090)),  # Walmart, Target
                     (sid(8347), sid(23112)),  # Exxon mobile, Chevron
                     (sid(24778), sid(18821)), # Sempra energy, Ventas
                     (sid(3496), sid(4521)),   # Home depot, Lowes
                     (sid(20088), sid(25006)), # Goldman Sachs, JP Morgan
                     (sid(4151), sid(5938)),   # Johnson and Johnson, Procter Gamble
                     (sid(6272), sid(23112)),  # PRAXAIR and CHEVRON    
                     (sid(863), sid(25165)),   # Billiton Limited ADS, Billiton p/c spons ADR
                     (sid(1638), sid(1637)),   # Comcact K, Comcast A
                     (sid(7784), sid(7767))    # Unilever NV, Unilever PLC
                ]

    context.spreads = []  
    
    context.coint_period = 30
    
    context.params = dict((pair, { "period_length": 30 }) for pair in context.stocks) 
    context.cointegrated = dict((pair, False) for pair in context.stocks)
    context.invested = dict((pair, False) for pair in context.stocks)



def handle_data(context, data):
    
    # Historical data of the stocks
    price_history = history(context.coint_period, '1d', 'price')
    
    # Get the current time
    time_list = ((str(get_datetime()).split(" "))[1]).split(":")
    (hour, minute) = (int(time_list[0]), int(time_list[1]))
    hour = hour - 5
    
    if (hour == 9 and minute == 31) or (hour == 16 and minute == 0):
    
        # Loop over all the stocks
        for pair in context.stocks:

            (context.stockX, context.stockY) = pair
    
            period_length = context.params[pair]["period_length"]
            x_price_hist = price_history[context.stockX][-period_length:]
            y_price_hist = price_history[context.stockY][-period_length:]
    
            # cointegration test first
        
            context.cointegrated[pair] = coint_test(x_price_hist , y_price_hist)
            
            # Current stock price
            stockXPrice = data.current(context.stockX, 'price')
            stockYPrice = data.current(context.stockY, 'price')
            
            # Difference between stock prices
            curDiff = stockXPrice - stockYPrice
    
            # Historical differences of the prices
            diff_hist = x_price_hist - y_price_hist
            mean = np.mean(diff_hist)
            std = np.std(diff_hist)
            
            # when the pairs are cointegrated, place order
            
            if context.cointegrated[pair] == True:
                log.info("Cointegrated!")
                place_orders(context, data, curDiff, mean, std)
                context.invested[pair] = True
                continue

def place_orders(context, data, curDiff, mean, std):  
    """ Buy when zscore is <= -2, sell when zscore >= 2"""  
    
    cash_per_trade = (context.portfolio.cash)/(2*max_trades)
    shares_x = int(cash_per_trade/data.current(context.stockX, 'price'))
    shares_y = int(cash_per_trade/data.current(context.stockY, 'price'))

    if curDiff >= mean + 2 * std:  
        log.info("Sell %s, Buy %s" % (context.stockX, context.stockY))

        order(context.stockX, -shares_x)  
        order(context.stockY, shares_y) 

        
    elif curDiff <= mean - 2 * std:
        log.info("Sell %s, Buy %s" % (context.stockY, context.stockX))

        order(context.stockX, shares_x)  
        order(context.stockY, -shares_y) 

#    elif (curDiff <= mean + 2 * std) or \
#         (curDiff >= mean - 2 * std):
#        log.info("Selling spread!")
#        sell_spread(context)  
#        

def sell_spread(context):  
    """  
    decrease exposure, regardless of posstockB_amountition long/short.  
    buy for a short position, sell for a long.  
    """  
    stockX_amount = context.portfolio.positions[context.stockX].amount  
    order(context.stockX, -stockX_amount)
    
    stockY_amount = context.portfolio.positions[context.stockY].amount  
    order(context.stockY, -stockY_amount)  
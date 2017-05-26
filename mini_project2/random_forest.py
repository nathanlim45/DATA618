from sklearn.ensemble import RandomForestClassifier
from collections import deque
import numpy as np

'''
Retruns 155.1%
Alpha 0.20
Sharpe 0.57
Drawdown -67.4%

The biggest difference from the sample model is that instead of comparing current value with yesterday's one, 
I have used a mean of 5 day historical data. I think this makes the model do more conservative investment. When I compare the two model, this gave me a much better result.'''


def initialize(context):
    context.security = sid(3149) # set the security
    context.window_length = 20 # number of bars or edges in a single decision tree
    
    # Use a random forest classifier
    context.classifier = RandomForestClassifier()
    
    context.recent_prices = deque(maxlen=context.window_length + 2)
    context.recent_value = deque(maxlen=context.window_length + 2)
    context.X = deque(maxlen=1000) # Independent variable
    context.Y = deque(maxlen=1000) # Dependent variable
    
    context.prediction = 0


def handle_data(context, data):
    context.recent_prices.append(data.current(context.security, 'price'))
    context.recent_value.append(data.current(context.security, 'price'))
    
    # Look back 5 days's historical prices
    price_change_hist = np.mean(data.history(context.security, "price", 5, "1d"))
    
    if len(context.recent_prices) == context.window_length + 2:
        
        # Make a list of 1's and 0's, 1 when the price increased from the prior bar
        changes =  context.recent_prices > price_change_hist
        values = np.array(context.recent_value).flatten()
        context.X.append(values[:-1]) # Add independent variables, the prior changes
        context.Y.append(changes) # Add dependent variable, the final change
        
        log.info(values[:-1]) #logging
        log.info(changes) #logging
        
        if len(context.Y) >= 300: # To make a good model increase number of data point
            
            context.classifier.fit(context.X, context.Y) # fit the data using the classifier
            
            context.prediction = context.classifier.predict(values[1:])
            
            # If prediction = 1, buy all shares affordable, if 0 sell all shares
            order_target_percent(context.security, context.prediction)
            
            record(prediction=int(context.prediction))




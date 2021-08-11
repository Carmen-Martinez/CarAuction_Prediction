# Used Car Selling Price Prediction

In this Web Application I utilize data science tools and techniques to predict used car selling prices at auctions.
 
I used Random Forest Regression to get my prediction with a 72% accuracy. 

Data: https://www.kaggle.com/tunguz/used-car-auction-prices
---------------------------------------------------------------------------------------------------------------------------------------------------------
## Repository Info

In this repository you will find the model I used and that you can download
[CarAuctionPriceFinal.pkl]

You can also find the original data file [car_prices.csv] + the cleaned date file [CarSalePrice.csv]

Finally you can find my source file [CarApp.py]

## Issues

I need to make it so that the body corresponds to the correct make/model. As of now, the
user has the ability to choose any body + make/model combination. 

Some of the features aren't affecting the prediction much, therefore I need to do further EDA + statistical analysis to check if there are any outliers
or low correlated features I must remove. 

Currently the 'odometer' feature affects the prediction the most. Perhaps it is too highly correlated? 


These are issues I hope to fix in the near future!



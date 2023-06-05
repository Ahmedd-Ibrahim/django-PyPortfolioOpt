from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

import os


# Create your views here.

def index(request):
    df = pd.read_csv(os.path.abspath(os.path.dirname(__file__))+"/data/stock_prices.csv", parse_dates=True, index_col="date")
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    # saves results to file weights.csv
    ef.save_weights_to_file("weights.csv")
    # print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)
    allocation, leftover = da.greedy_portfolio()
    #print on console
    print("Discrete allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))

    return HttpResponse("Funds remaining: ${:.2f}".format(leftover))


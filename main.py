import numpy as np
import cvxpy as cp
import random
import fisherMarket as m
import test.fisherVerifier as fv
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
#from matplotlib import pyplot

# Set the number of bidders that we want
numbidders = 2

# Set the number of goods that we want
numgoods = 2

# Set the number of trials that we want
numtrials = 20

allweights = [0 for i in range(numtrials)]
allbeliefs = [0 for i in range(numtrials)]
allaggregates = [0 for i in range(numtrials)]

for i in range(numtrials):

    # Create random valuations for our bidders
    # Matrix of valuations: |buyers| x |goods|
    unnormal_valuations = np.array([[random.random() for j in range(numgoods)] for i in range(numbidders)])
    valuations = (unnormal_valuations.T /np.sum(unnormal_valuations, axis = 1)).T
    print("valuations: ", valuations)

    # Budgets of buyers: |buyers|
    # Note: these must be normalized (add to 1)
    first_step = [random.random() for i in range(numbidders - 1)]
    first_step.sort()
    second_step = [0] + first_step + [1]
    third_step = [second_step[i + 1] - second_step[i] for i in range(len(second_step) - 1)]
    budgets = np.array(third_step)
    print("budgets", budgets)

    # Create Market
    market = m.FisherMarket(valuations, budgets)

    # Solve Market
    X, p = market.solveMarket("cobb-douglas", printResults=True)

    fv.verify(X, p, valuations, budgets, utility = "cobb-douglas")

    # Now, we have 'valuations' as our valuations, 'budgets' as our budgets, and 'p' as our prices

    # Switching over to the QA pooling context, we now relabel budgets as weights, valuations as beliefs,
    # and prices as aggregates

    allweights[i] = budgets
    allbeliefs[i] = valuations
    allaggregates[i] = p

print("allweights", allweights)
print("allbeliefs", allbeliefs)
print("allaggregates", allaggregates)
import math
import numpy as np


def sigmoid(N):
    return np.vectorize(lambda x: 1 / (1 + math.exp(-x)))(N)


def predict(features, weights):
    '''
    :param features: (M, N)
    :param weights: (M, K)
    :return: 1D array of probabilities
    '''
    z = np.dot(features, weights)
    return sigmoid(z)


def cost_function(features, weights, labels):
    '''
    :param features: (M, N)
    :param weights: (M, N)
    :param labels: (M, 1)
    :return: Mean cost
    Cost = (-labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)
    predictions = predict(features, weights)

    class1_cost = -labels * np.log(predictions)
    class2_cost = -(1 - labels) * np.log(1 - predictions)

    cost = class1_cost + class2_cost

    return cost.sum() / observations


def update_weights(features, weights, labels, lr):
    '''
    Vectorized Gradient descent
    :param features: (M, N)
    :param weights: (M, 1)
    :param labels: (M, K)
    :param lr: float
    :return: average gradient
    '''
    N = len(features)

    # 1 - Get Predictions
    predictions = predict(features, weights)

    # 2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T, predictions - labels)

    # 3 Take the average cost derivative for each feature
    gradient /= N

    # 4 - Multiply the gradient by our learning rate
    gradient *= lr

    # 5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

#
# def decision_boundary(prob):
#     return 1 if prob >= .5 else 0


def classify(predictions):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(lambda p: 1 if p >= .5 else 0)
    return decision_boundary(predictions).flatten()


def train(features, weights, labels, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, weights, labels, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, weights, labels)
        cost_history.append(cost)

        # Log Progress
        # if i % 100 == 0:
        #     print("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history


features = np.array([[4.85, 9.63],
                     [8.62, 3.23],
                     [5.43, 8.23],
                     [9.21, 6.34]])
labels = np.array([[1], [0], [1], [0]])
weights = np.array([[0.0], [0.0]])

final_weights, cost_history = train(features, weights, labels, 0.001, 10000)
print('Final Weights : ' + str(final_weights))

predictions = predict(features, final_weights)
results = classify(predictions)

print("Probabilities : " + str(predictions))
print("classifications : " + str(results))

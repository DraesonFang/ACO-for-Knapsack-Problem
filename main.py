'''
ECMM409 – Nature-Inspired Computation
COURSEWORK
'''
from docutils.nodes import label
from holoviews.plotting.bokeh.styles import marker
from sympy.stats.rv import probability

from Setting import security_van_capacity, population_size, iters, evaporation_rate, alpha, beta, random_seed, pheromone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bag_data = []

# template parameter,temp is used to store the value of capacity,weight,value etc,tempDict is used to store a dictionary include weight and value
temp = 0
tempDict = {}

# load data from txt document,and put them into bag_data
with open('BankProblem.txt', 'r') as f:
    for line in f:
        line = line.strip()
        # print(line)
        if line.startswith('security van capacity'):
            temp = int((line.split(':')[1]).strip())
            # print(temp)
            security_van_capacity = temp
        elif line.startswith('bag'):
            tempDict = {"weight": 0, "value": 0}
        elif line.startswith('weight'):
            temp = float((line.split(':')[1]).strip())
            if 'weight' in tempDict:
                tempDict['weight'] = temp
        elif line.startswith('value'):
            temp = float((line.split(':')[1]).strip())
            if 'value' in tempDict:
                tempDict['value'] = temp
            bag_data.append(tempDict)
    # print(bag_data)

data = pd.DataFrame(bag_data)

# create a value matrix in shape (10,10)
value_matrix = np.array(data[['value']].values.reshape(100))
# create a weight matrix in shape (10,10)
weight_matrix = np.array(data[['weight']].values.reshape(100))

value_weight_ratio = value_matrix / weight_matrix


def chose_path(alpha, beta, heru_matric, phrm_matrix, bag):
    heru_matric[bag] = 0
    phrm_matrix[bag] = 0
    total_value = np.array([phrm_matrix[i] ** alpha * heru_matric[i] ** beta for i in range(len(phrm_matrix))]).sum()
    probability_matrix = np.array(
        [phrm_matrix[i] ** alpha * heru_matric[i] ** beta / total_value for i in range(len(phrm_matrix))])
    probability_matrix = np.cumsum(probability_matrix)
    np.random.seed(None)
    choice_seed = np.random.rand()
    for i, _ in enumerate(probability_matrix):
        if (choice_seed <= probability_matrix[i]) and (i not in select_bag):
            return i


# select final bag
def chose_final_path(alpha, beta, heru_matric, phrm_matrix, temp_weight, bag):
    heru_matric[bag] = 0
    phrm_matrix[bag] = 0
    heru_matric[weight_matrix > temp_weight] = 0
    phrm_matrix[weight_matrix > temp_weight] = 0
    total_values = np.array([phrm_matrix[i] ** alpha * heru_matric[i] ** beta for i in range(len(phrm_matrix))]).sum()
    probability_matrix = np.array(
        [phrm_matrix[i] ** alpha * heru_matric[i] ** beta / total_values for i in range(len(phrm_matrix))])
    probability_matrix = np.cumsum(probability_matrix)
    np.random.seed(None)
    choice_seed = np.random.rand()
    for i, _ in enumerate(probability_matrix):
        if (choice_seed <= probability_matrix[i]) and (i not in select_bag):
            return i

    return -1


# return a total value
def fitness(slct_bag):
    total_val = np.array([value_matrix[i] for i in slct_bag]).sum()

    return total_val


def update_pheromone(bag, fitness_val, p_m, mode=1):
    if mode == 1:
        t_max = 5.0
        t_min = 0.1
    else:
        t_max = 10.0
        t_min = 0.01

    p_m[bag] += fitness_val/500

    return np.clip(p_m, t_min, t_max)


def evaporate_pheromone():
    for _ in range(len(pheromone_matrix)):
        pheromone_matrix[_] *= (1-evaporation_rate)


'''
Parameter Defination
'''
select_bag = np.array([])
total_value = 0
total_weight = 0

global_best_bags = np.array([])
global_best_value = 0

clct_value = np.array([])
clct_indicator = np.array([])
best_ant_ind = 0

heuristic_matrix = np.array([round((value_matrix[i] / weight_matrix[i]), 4) for i in range(100)])

# create a random seed to initialise pheromone matrix
np.random.seed(random_seed)
p_seed = np.random.rand()

# create a pheromone matrix filling with random seed in initialization
pheromone_matrix = np.full(100, p_seed)
for _ in range(iters):
    iter_best_bag = np.array([])
    iter_best_value = 0

    for ant in range(population_size):
        select_bag = np.array([], dtype=int)
        total_value = 0
        total_weight = 0

        while True:
            if total_weight > security_van_capacity:
                select_bag = select_bag[:-1]
                break
            #choose a bag
            ind = chose_path(alpha, beta, heuristic_matrix.copy(), pheromone_matrix.copy(), select_bag)
            # if ind equal -1, which mean there is no bag that is selected
            if ind == -1:
                break

            select_bag = np.append(select_bag, ind)
            total_weight = weight_matrix[select_bag].sum()

        # see if total wight bigger than van can carry
        total_weight = weight_matrix[select_bag].sum()
        temp_weight = security_van_capacity - total_weight

        # check if there is any bag that van can carry
        while temp_weight >= 1:
            ind = chose_final_path(alpha, beta, heuristic_matrix.copy(), pheromone_matrix.copy(), temp_weight,
                                   select_bag)
            if ind == -1:
                break
            select_bag = np.append(select_bag, ind)
            total_weight = weight_matrix[select_bag].sum()
            temp_weight = security_van_capacity - total_weight

        total_value = fitness(select_bag)
        total_weight = weight_matrix[select_bag].sum()
        # iteration best value
        if iter_best_value < total_value:
            iter_best_value = total_value
            iter_best_bag = select_bag
        # Global best value
        if global_best_value <= total_value:
            global_best_value = total_value
            global_best_bags = select_bag
            best_ant_ind = _
        clct_value = np.append(clct_value, total_value)
        clct_indicator = np.append(clct_indicator, round(iter_best_value / weight_matrix[iter_best_bag].sum(), 4))

        # Pheromone update with iteration-best or global-best
        if _ < iters / 2:
            # Use iteration-best update in the first half of the iterations
            # for _ in range(population_size):
            pheromone_matrix = update_pheromone(iter_best_bag, iter_best_value, pheromone_matrix.copy(), 1)
        else:
            # Use global-best update in the second half of the iterations
            # for _ in range(population_size):
            pheromone_matrix = update_pheromone(global_best_bags, global_best_value, pheromone_matrix.copy(), 2)

        evaporate_pheromone()

print(global_best_bags)
print(global_best_value)
print(weight_matrix[global_best_bags].sum())
print(round(global_best_value / weight_matrix[global_best_bags].sum(), 4))
print(select_bag)
print(total_value)
print(total_weight)
print(total_value / total_weight)

# visualisation
fig, ax = plt.subplots(2, 1, figure=(16, 10))
ax[0].plot(range(len(clct_value)), clct_value, c='r', label='Total Value')
ax[0].scatter(best_ant_ind, global_best_value, c='g', label='Best Value')
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Total value")
ax[0].set_title("Best Total value in each iteration")
ax[0].legend()
ax[1].plot(range(len(clct_indicator)), clct_indicator, c='b', label='Unit weight value')
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Best value-to-weight-ratio")
ax[1].set_title("Best Unit weight value in each iteration")
ax[1].legend()
plt.subplots_adjust(wspace=0.5, hspace=0.7)
plt.show()

'''
ECMM409 – Nature-Inspired Computation
COURSEWORK
'''
from sympy.stats.rv import probability

from Setting import security_van_capacity, population_size, pheromone, evaporation_rate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bag_data = []

#template parameter,temp is used to store the value of capacity,weight,value etc,tempDict is used to store a dictionary include weight and value
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
# print(data)


# create a value matrix in shape (10,10)
value_matrix = np.array(data[['value']].values.reshape(100))
# create a weight matrix in shape (10,10)
weight_matrix = np.array(data[['weight']].values.reshape(100))

# print(value_matrix)
# print(weight_matrix)
# create a heuristic matrix, base on inverse of value / weight, which means the smaller the value the better the characterisation
heuristic_matrix = np.array([round(1/value_matrix[i]/weight_matrix[i],4) for i in range(100)])
# print(heuristic_matrix)

# create a random seed to initialise pheromone matrix
# np.random.seed(77)
p_seed = np.random.rand()

# create a pheromone matrix filling with random seed in initialization
pheromone_matrix = np.full(100,p_seed)

# print(pheromone_matrix)
# remove node
# using log to cumsum
def chose_path(alpha,beta,heru_matric,phrm_matrix,bag):
    heru_matric[bag] = 0
    phrm_matrix[bag] = 0
    total_value = np.array([phrm_matrix[i]**alpha*heru_matric[i]**beta for i in range(len(phrm_matrix))]).sum()
    probability_matrix = np.array([phrm_matrix[i]**alpha*heru_matric[i]**beta / total_value for i in range(len(phrm_matrix))])
    probability_matrix = np.cumsum(probability_matrix)
    choice_seed = np.random.rand()
    for i,_ in enumerate(probability_matrix):
        if (choice_seed <= probability_matrix[i]) and (i not in select_bag):
            return i

# return a total value
def fitness(slct_bag):
    total_value = np.array([value_matrix[i] for i in slct_bag]).sum()
    total_weight = np.array([weight_matrix[i] for i in slct_bag]).sum()
    return total_value if total_weight < security_van_capacity else -1

def update_phremone(ind,bags):
    pheromone_matrix[ind] += fitness(bags)
    for _ in range(len(pheromone_matrix)):
        pheromone_matrix[_] *= evaporation_rate

# print(value_matrix[value_matrix>security_van_capacity])

'''
Parameter Defination
'''
select_bag = np.array([])
total_value = 0
num_bags = 100
num_ants = 20
alpha = 1
beta = 2
for ant in range(num_ants):
    select_bag = []

    total_value = 0
    total_weight = 0
    while True:
        if total_weight> security_van_capacity:
            select_bag = select_bag[:-1]
            break
        # print(pheromone_matrix)
        ind = chose_path(alpha,beta,heuristic_matrix.copy(),pheromone_matrix.copy(),select_bag)
        if ind is None:
            break
        select_bag.append(ind)
        total_value = fitness(select_bag)
        total_weight = weight_matrix[select_bag].sum()
        update_phremone(ind,select_bag)

    print('*'*100)
    total_value = value_matrix[select_bag].sum()
    print(select_bag)
    print(total_value)
    total_weight = weight_matrix[select_bag].sum()
    print(total_weight)
    print(security_van_capacity-total_weight)
    print(round(total_value/total_weight,4))
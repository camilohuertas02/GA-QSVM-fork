import typing
import qiskit
import random
import numpy as np
from ..backend import utilities

import numpy as np
import qiskit.quantum_info as qi

def create_params(depths, num_circuits, num_generations):
    # zip num_generations, depths, num_circuits as params
    params = []
    for num_generation in num_generations:
        for depth in depths:
            for num_circuit in num_circuits:
                params.append((depth, num_circuit, num_generation))
    return params        

def calculate_risk(utests, V):
    num_qubits = V.num_qubits
    # Create |0> state
    zero_state = np.zeros((2**num_qubits, 1))
    zero_state[0] = 1
    # Create |0><0| matrix
    zero_zero_dagger = np.outer(zero_state, np.conj(zero_state.T))
    V_matrix = qi.DensityMatrix(V).data
    risk = []
    for utest in utests:
        Ui_matrix = qi.DensityMatrix(utest).data
        # Eq inside L1 norm of matrix ^2
        eq = (Ui_matrix @ zero_zero_dagger @ np.conj(Ui_matrix.T) - V_matrix @ zero_zero_dagger @ np.conj(V_matrix.T))
        # L1 norm of matrix ^ 2
        risk.append(np.linalg.norm(eq, 1)**2)
    # Expected risk / 4
    return np.mean(risk)/4  

def fight(population):
    circuits = random.sample(population, 2)
    return circuits[0] if circuits[0].fitness > circuits[1].fitness else circuits[1]


def random_mutate(population, prob, mutate_func):
    random_circuit_index = np.random.randint(0, len(population))
    random_value = random.random()
    if random_value < prob:
        print(f'Mutate {random_circuit_index}')
        population[random_circuit_index].mutate(mutate_func)
    return population


def calculate_strength_point(self):
    inverse_fitnesss = [1 - circuit.fitness for circuit in self.population]
    mean_inverse_fitnesss = np.mean(inverse_fitnesss)
    std_inverse_fitnesss = np.std(inverse_fitnesss)
    strength_points = [(1 - circuit.fitness - mean_inverse_fitnesss) /
                        std_inverse_fitnesss for circuit in self.population]
    scaled_strength_points = utilities.softmax(strength_points, self.depth)
    for i, circuit in enumerate(self.population):
        circuit.strength_point = scaled_strength_points[i]
    return



def sort_by_fitness(objects: list, fitnesss: list):
    if not isinstance(fitnesss[0], (list, tuple)):
        print(">> Ordenando por Accuracy máximo")
        combined_list = list(zip(objects, fitnesss))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_combined_list]
    print(">> Ordenando por frente de pareto (modo 1) ")
    n = len(objects)
    domination_counts = [0] * n
    dominated_lists = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        acc_i, depth_i = fitnesss[i][0], fitnesss[i][1]
        for j in range(n):
            if i == j:
                continue
            acc_j, depth_j = fitnesss[j][0], fitnesss[j][1]
            
            if (acc_i >= acc_j and depth_i <= depth_j) and (acc_i > acc_j or depth_i < depth_j):
                dominated_lists[i].append(j)
            elif (acc_j >= acc_i and depth_j <= depth_i) and (acc_j > acc_i or depth_j < depth_i):
                domination_counts[i] += 1
        
        if domination_counts[i] == 0:
            fronts[0].append(i)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for idx in fronts[i]:
            for dominated_idx in dominated_lists[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        i += 1
        fronts.append(next_front)

    sorted_objects = []
    for front in fronts:
        if not front:
            continue
        front_sorted = sorted(front, key=lambda idx: fitnesss[idx][0], reverse=True)
        for idx in front_sorted:
            sorted_objects.append(objects[idx])

    return sorted_objects

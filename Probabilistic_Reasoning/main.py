# main.py: Probabilistic Reasoning Mini-Project
# Author: Chen
# Description: Implements Bayesian network inference using enumeration and variable elimination, simulates dynamic BNs, and computes utilities in decision networks for student success modeling.

import hashlib, numpy as np, random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

id = input()

idHash = int(hashlib.sha256(id.encode()).hexdigest(), 16) % (2**31)

np.random.seed(idHash)
random.seed(idHash)

print(f"Reproducibility seed is: {idHash}")

variables = {
    'H': ['healthy', 'sick'],
    'S': ['good', 'poor'],
    'A': ['high', 'low'],
    'T': ['yes', 'no'],
    'E': ['well', 'badly'],
    'P': ['pass', 'fail']
}

def randomCpt(var, parents):
    domain = [variables[p] for p in parents]
    assignments = np.array(np.meshgrid(*domain)).T.reshape(-1, len(parents))
    table = {}
    for parentVals in assignments:
        key = tuple(parentVals)
        vals = np.random.rand(len(variables[var]))
        vals = vals / vals.sum() 
        table[key] = vals
    return table

cpts = {}

cpts['H'] = np.random.rand(2)
cpts['H'] = cpts['H'] / cpts['H'].sum()

cpts['T'] = np.random.rand(2)
cpts['T'] = cpts['T'] / cpts['T'].sum()

cpts['S'] = randomCpt('S', ['H'])
cpts['A'] = randomCpt('A', ['S'])
cpts['E'] = randomCpt('E', ['A', 'T'])
cpts['P'] = randomCpt('P', ['E', 'H'])

print("Bayesian Network CPTs initialized with random but reproducible probabilities.")

def enumerateAll(vars, evidence, cpts):
    if not vars:
        return 1.0
    Y = vars[0]
    rest = vars[1:]

    parentsMap = {
        'H': [],
        'T': [],
        'S': ['H'],
        'A': ['S'],
        'E': ['A', 'T'],
        'P': ['E', 'H']
    }

    def localProb(var, val, ev):
        if parentsMap[var]:
            parentVals = tuple(ev[p] for p in parentsMap[var])
            probs = cpts[var][parentVals]
        else:
            probs = cpts[var]
        return probs[variables[var].index(val)]

    if Y in evidence:
        yVal = evidence[Y]
        return localProb(Y, yVal, evidence) * enumerateAll(rest, evidence, cpts)
    else:
        total = 0.0
        for yVal in variables[Y]:
            newEv = evidence.copy()
            newEv[Y] = yVal
            total += localProb(Y, yVal, newEv) * enumerateAll(rest, newEv, cpts)
        return total

def enumerationAsk(X, evidence, cpts):
    Q = {}
    for xVal in variables[X]:
        extendedEv = evidence.copy()
        extendedEv[X] = xVal
        Q[xVal] = enumerateAll(list(variables.keys()), extendedEv, cpts)
    total = sum(Q.values())
    return {k: v / total for k, v in Q.items()}

class Factor:
    def __init__(self, vars, table):
        self.vars = vars
        self.table = table 

    def restrict(self, var, value):
        varIdx = self.vars.index(var)
        newTable = [(assign, prob) for assign, prob in self.table if assign[varIdx] == value]
        newVars = [v for v in self.vars if v != var]
        return Factor(newVars, newTable)

    def multiply(self, other):
        common = set(self.vars) & set(other.vars)
        selfIdx = {v: i for i, v in enumerate(self.vars)}
        otherIdx = {v: i for i, v in enumerate(other.vars)}
        newVars = list(set(self.vars) | set(other.vars))
        newTable = []
        for sAssign, sProb in self.table:
            for oAssign, oProb in other.table:
                if all(sAssign[selfIdx[v]] == oAssign[otherIdx[v]] for v in common):
                    newAssign = tuple(sAssign[selfIdx.get(v, -1)] if v in self.vars else oAssign[otherIdx[v]] for v in newVars)
                    newTable.append((newAssign, sProb * oProb))
        return Factor(newVars, newTable)

    def sumOut(self, var):
        varIdx = self.vars.index(var)
        newVars = [v for v in self.vars if v != var]
        summed = defaultdict(float)
        for assign, prob in self.table:
            newAssign = tuple(assign[i] for i in range(len(assign)) if i != varIdx)
            summed[newAssign] += prob
        return Factor(newVars, list(summed.items()))

def variableElimination(queryVar, evidence, cpts, elimOrder):
    factors = []
    parentsMap = {
        'H': [],
        'T': [],
        'S': ['H'],
        'A': ['S'],
        'E': ['A', 'T'],
        'P': ['E', 'H']
    }
    for var, cpt in cpts.items():
        if isinstance(cpt, np.ndarray):
            vars_ = [var]
            table = [( (variables[var][val],), prob ) for val, prob in enumerate(cpt)]
        else:
            vars_ = parentsMap[var] + [var]
            table = []
            for parent_vals, child_probs in cpt.items():
                for i, prob in enumerate(child_probs):
                    child_val = variables[var][i]
                    assignment = parent_vals + (child_val,) 
                    table.append((assignment, prob))
        
        factor = Factor(vars_, table)
        if var in evidence:
            factor = factor.restrict(var, evidence[var])
        factors.append(factor)

    for var in elimOrder:
        if var != queryVar and var not in evidence:
            relevant = [f for f in factors if var in f.vars]
            factors = [f for f in factors if var not in f.vars]
            if relevant:
                product = relevant[0]
                for f in relevant[1:]:
                    product = product.multiply(f)
                summed = product.sumOut(var)
                factors.append(summed)

    if factors:
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        total = sum(prob for _, prob in result.table)
        queryIdx = result.vars.index(queryVar)
        return {assign[queryIdx]: prob / total for assign, prob in result.table}
    return {}

def dbnVariableElim(queryVar, evidence, cpts, elimOrder):
    factors = []
    parentsMap = {
        'S1': ['StudyHours'], 
        'H1': ['S1'],
        'P1': ['H1'],
    }
    
    for var, cpt in cpts.items():
        if var in parentsMap and parentsMap[var]:
            vars_ = parentsMap[var] + [var] 
            table = [(assign, prob) for assign, prob in cpt.items()]
        else:
            vars_ = [var]
            table = [((k[0],), prob) for k, prob in cpt.items()] 
            
        factor = Factor(vars_, table)
        if var in evidence:
            factor = factor.restrict(var, evidence[var])
        factors.append(factor)


    for var in elimOrder:
        if var != queryVar and var not in evidence:
            relevant = [f for f in factors if var in f.vars]
            factors = [f for f in factors if var not in f.vars]
            if relevant:
                product = relevant[0]
                for f in relevant[1:]:
                    product = product.multiply(f)
                summed = product.sumOut(var)
                factors.append(summed)

    if factors:
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        total = sum(prob for _, prob in result.table)
        return {assign[result.vars.index(queryVar)]: prob / total for assign, prob in result.table}
    return {}

cpts = {
    'S1': {('low', 'low'): 0.3, ('low', 'high'): 0.7,
           ('medium', 'low'): 0.5, ('medium', 'high'): 0.5,
           ('high', 'low'): 0.8, ('high', 'high'): 0.2},
    'H1': {('low', 'sick'): 0.7, ('low', 'healthy'): 0.3,
           ('high', 'sick'): 0.2, ('high', 'healthy'): 0.8},
    'P1': {('sick', 'low'): 0.9, ('sick', 'high'): 0.1,
           ('healthy', 'low'): 0.2, ('healthy', 'high'): 0.8},
}

utilityTable = {
    ('low',): 20,
    ('high',): 100,
}

def computeExpUtil(cpts, utilityTable, decisionVals):
    result = {}
    cptS1 = cpts['S1'] 
    cptH1 = cpts['H1'] 
    cptP1 = cpts['P1'] 
    s1Vals = ['low', 'high']
    h1Vals = ['sick', 'healthy']
    p1Vals = ['low', 'high']
    for d in decisionVals:
        expUtil = 0.0
        for s1 in s1Vals:
            for h1 in h1Vals:
                for p1 in p1Vals:
                    probS1 = cptS1.get((d, s1), 0.0)
                    probH1 = cptH1.get((s1, h1), 0.0)
                    probP1 = cptP1.get((h1, p1), 0.0)
                    jointProb = probS1 * probH1 * probP1
                    if jointProb == 0.0:
                        continue            
                    utility = utilityTable.get((p1,), 0.0)
                    expUtil += jointProb * utility
        result[d] = expUtil
    return result

decisionVals = ['low', 'medium', 'high']
expUtils = computeExpUtil(cpts, utilityTable, decisionVals)
print("Expected Utilities:", expUtils)
from scipy.optimize import minimize, least_squares
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from physicalParameters import get_parametersValues

n_hardcoded, A_hardcoded, a, dhdx, h_obs, rho, g, x = get_parametersValues()
observation_size = len(h_obs)

def objective_function_n(n):
    h_theoretical = np.zeros(observation_size)
    delta_h = np.zeros(observation_size)
    sum = 0
    for i in range(observation_size):
        numerator = a[i]*(n+2)
        denominator = 2*A_hardcoded*((rho*g)**n)*dhdx[i]*((abs(dhdx[i]))**(n-1))
        h_theoretical[i] = (-(numerator/denominator))**(1/(n+2))
        delta_h[i] = h_theoretical[i] - h_obs[i]
        sum += delta_h[i]**2
    return sum

def objective_function_A(A):
    h_theoretical = np.zeros(observation_size)
    delta_h = np.zeros(observation_size)
    sum = 0
    for i in range(observation_size):
        numerator = a[i]*(n_hardcoded+2)
        denominator = 2*A*((rho*g)**n_hardcoded)*dhdx[i]*((abs(dhdx[i]))**(n_hardcoded-1))
        h_theoretical[i] = (-(numerator/denominator))**(1/(n_hardcoded+2))
        delta_h[i] = h_theoretical[i] - h_obs[i]
        sum += delta_h[i]**2
    return sum

def objective_function_multivariate(X):
    h_theoretical = np.zeros(observation_size)
    delta_h = np.zeros(observation_size)
    sum = 0
    for i in range(observation_size):
        numerator = a[i]*(X[0]+2)
        denominator = 2*X[1]*((rho*g)**X[0])*dhdx[i]*((abs(dhdx[i]))**(X[0]-1))
        h_theoretical[i] = (-(numerator/denominator))**(1/(X[0]+2))
        delta_h[i] = h_theoretical[i] - h_obs[i]
        sum += delta_h[i]**2
    return sum

def plot_functionValue_n(fun_value, n_value):
    array_n = np.arange(start=(n_value-0.05),stop=(n_value+0.055),step=0.005)
    function_values = np.zeros(len(array_n))
    for j,n_optimal in enumerate(array_n):
        function_values[j] = objective_function_n(n_optimal)
    plt.plot(array_n,function_values,'r--o')
    plt.plot(n_value,fun_value,'go')
    plt.title('Function value x $n$')
    plt.xlabel('$n$')
    plt.ylabel('Function value')
    plt.show()

def plot_functionValue_A(fun_value, A_value):
    array_A = np.arange(start=(A_value/5),stop=(A_value/(0.5)),step=(0.01*(10**(-27))))
    print(len(array_A))
    function_values = np.zeros(len(array_A))
    for j,A_optimal in enumerate(array_A):
        function_values[j] = objective_function_A(A_optimal)
    plt.plot(array_A,function_values,'r--o')
    plt.plot(A_value,fun_value,'go')
    plt.title('Function value x $A$')
    plt.xlabel('$A$')
    plt.ylabel('Function value')
    plt.show()

def plot_functionValue_multivariate(fun_value, array_multivariate):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n = np.arange(start=(array_multivariate[0]-0.01),stop=(array_multivariate[0]+0.011),step=0.001)
    A = np.arange(start=(array_multivariate[1]-(1*(10**(-31)))),stop=(array_multivariate[1]+(1*(10**(-31)))),step=((5)*(10**(-34))))
    n, A = np.meshgrid(n, A)
    function_values = np.zeros(n.shape)
    for xn in range(n.shape[0]):
        for xy in range(n.shape[1]):
            input = [n[xn,xy],A[xn,xy]]
            function_values[xn,xy] = objective_function_multivariate(input)
    surf = ax.plot_surface(n, A, function_values, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()

def plot_thickness_multivariate(array_multivariate):
    X = array_multivariate
    h_theoretical = np.zeros(observation_size)
    for i in range(observation_size):
        numerator = a[i]*(X[0]+2)
        denominator = 2*X[1]*((rho*g)**X[0])*dhdx[i]*((abs(dhdx[i]))**(X[0]-1))
        h_theoretical[i] = (-(numerator/denominator))**(1/(X[0]+2))
    plt.plot(x,h_obs,'r',label='Observed')
    plt.plot(x,h_theoretical,'g--',label='Computed')
    plt.title('Thickness vs $x$ (n = ' + str(X[0]) + ' and A = ' + str(X[1]) + ')')
    plt.xlabel('$x$')
    plt.legend()
    plt.ylabel('Thickness value')
    plt.show()

res_task1 = minimize(objective_function_n, [2.5], method='Nelder-Mead', options={'disp': True})
#plot_functionValue_n(res.fun,res.x)

res_task4 = minimize(objective_function_A, [10**(-28)], method='Nelder-Mead', options={'disp': True})
#plot_functionValue_n(res.fun,res.x)

vector_variables = [n_hardcoded, A_hardcoded]
res_task6 = minimize(objective_function_multivariate, vector_variables, method='Nelder-Mead', options={'disp': True, 'maxiter': 500, 'maxfev': 1000})
#plot_functionValue_multivariate(res.fun, res.x)
#plot_thickness_multivariate(res.x)
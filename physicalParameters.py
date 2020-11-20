import pandas as pd
import numpy as np

def get_hardCoded_variableValues():

    #Hard-coded values:

    n = 3 #flow law exponent
    A = 10**(-25) #rate factor
    rho = 918 #density of the ice (kg/m3)
    g = 3.710000 #gravitational acceleration on Mars (m/s2)

    return (n, A, rho, g)

def read_shallowRadar_data():

    sharad_data = pd.read_csv('Sharad.csv') #processed data in .csv format

    a = np.array(sharad_data['a']).reshape(-1,1) #mass balance function values
    dhdx = np.array(sharad_data['dhdx']).reshape(-1,1) #slope of the ice surface measurements
    h_obs = np.array(sharad_data['H_obs']).reshape(-1,1) #thickness of ice surface measurements
    x = np.array(sharad_data['x']).reshape(-1,1) #x positions
    
    return(a, dhdx, h_obs, x)

def get_parametersValues():

    (n, A, rho, g) = get_hardCoded_variableValues()
    (a, dhdx, h_obs, x) = read_shallowRadar_data()

    return(n, A, a, dhdx, h_obs, rho, g, x)
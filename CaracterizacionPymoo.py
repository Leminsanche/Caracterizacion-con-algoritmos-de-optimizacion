import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.optimize import minimize


def Neo_Hooke(landa, C):
    c1 = C[0]
    sigma = 2*c1*((landa**2)-(1/landa))    
    return sigma

def Mooney_Rivlin(landa, C):
    c1 = C[0]
    c2 = C[1]
    sigma = 2*(c1+(c2/landa))*((landa**2)-(1/landa))
    return sigma

def Demiray(landa,C):
    c1 = C[0]
    c2 = C[1]
    aux = (c2/2)*((landa**2)+(2/landa)-3)
    sigma = c1*np.e**(aux)*((landa**2)-(1/landa))
    return sigma
    
def Yeoh(landa,C):
    c1 = C[0]
    c2 = C[1]
    c3 = C[2]
    I1 = landa**2 + (2/landa)
    aux = c1+2*c2*(I1-3) + 3*c3*(I1-3)**2
    sigma = 2*aux*((landa**2)-(1/landa))
    return sigma


############ Definicion del problema y su clase respectiva
class MyProblem(ElementwiseProblem):

    def __init__(self,datos_x,datos_y,Modelo,num_const):
        
        self.datos_x = datos_x
        self.datos_y = datos_y
        self.Modelo = Modelo
        self.num_const = num_const
        super().__init__(n_var=num_const,
                         n_obj=1,
                         n_ieq_constr=1,
                         xl=-2,
                         xu=2 )

    def _evaluate(self, x, out, *args, **kwargs):
        aux = self.Modelo(self.datos_x, x)
        f1 = np.linalg.norm(self.datos_y - aux)

        g1 = x[0] > 0

        out["F"] = f1
        out["G"] = g1




datos = np.loadtxt('Promedio Hipoxia.txt')
datos_x = datos[:,0]
datos_y = datos[:,1]


problem1 = MyProblem(datos_x,datos_y,Neo_Hooke,1)
problem2 = MyProblem(datos_x,datos_y,Mooney_Rivlin,2)
problem3 = MyProblem(datos_x,datos_y,Demiray,2)
problem4 = MyProblem(datos_x,datos_y,Yeoh,3)


## algoritmo genetico ##
algorithm = GA(
    pop_size=1000,
    eliminate_duplicates=True)

res = minimize(problem4,
               algorithm,
               verbose=False)

print("Best solution found in GA: \nX = %s\nF = %s" % (res.X, res.F))


###### Diferenttial Evolution ####

algorithm = DE(
    pop_size=1000,
    sampling=LHS(),
    variant="DE/rand/1/bin",
    CR=0.3,
    dither="vector",
    jitter=False
)

res = minimize(problem4,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found DE : \nX = %s\nF = %s" % (res.X, res.F))



algorithm = ES(n_offsprings=200, rule=1.0 / 7.0)

res = minimize(problem4,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=False)

print("Best solution found ES : \nX = %s\nF = %s" % (res.X, res.F))




algorithm = SRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

res = minimize(problem4,
               algorithm,
               ("n_gen", 100),
               seed=1,
               verbose=False)

print("Best solution found SRES : \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))

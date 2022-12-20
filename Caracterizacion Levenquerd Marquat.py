import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model


def Neo_Hooke(landa, c1):
    sigma = 2*c1*((landa**2)-(1/landa))    
    return sigma

def Mooney_Rivlin(landa, c1,c2):
    sigma = 2*(c1+(c2/landa))*((landa**2)-(1/landa))
    return sigma

def Demiray(landa,c1,c2):
    aux = (c2/2)*((landa**2)+(2/landa)-3)
    sigma = c1*np.e**(aux)*((landa**2)-(1/landa))
    return sigma
    
def Yeoh(landa,c1,c2,c3):
    I1 = landa**2 + (2/landa)
    aux = c1+2*c2*(I1-3) + 3*c3*(I1-3)**2
    sigma = 2*aux*((landa**2)-(1/landa))
    return sigma

#Modelos usados
Modelos = [[Neo_Hooke,1,'Neo_Hooke'],[Mooney_Rivlin,2,'Mooney_Rivlin'],[Demiray,3,'Demiray'],[Yeoh,4,'Yeoh']]


#Exportar Datos

Datos_Control = np.loadtxt('Promedio Control.txt') 
Datos_Hipoxia = np.loadtxt('Promedio Hipoxia.txt')


landa_control = Datos_Control[:,0]
landa_hipoxia = Datos_Hipoxia[:,0]

sigma_control = Datos_Control[:,1]
sigma_hipoxia = Datos_Hipoxia[:,1]


plt.plot(landa_control,sigma_control,'k',label = 'Datos Control')
plt.plot(landa_hipoxia,sigma_hipoxia,'g',label = 'Datos Hipoxia')
plt.grid()
plt.legend()
plt.show()




###################################### Caracterizacion LM Datos Control #############################################3
######################################################################################################################



x  , y = landa_control,sigma_control

for i in Modelos:
    gmodel = Model(i[0])
    
    if i[1] == 1:
        
        result = gmodel.fit(y, landa=x,c1 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2= 1 - rss/tss
        #print(f"R² = {1 - rss/tss:.3f}")
        
    elif i[1] == 2:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
        
    elif i[1] == 3:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
    

    elif i[1] == 4:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1,c3 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
        
    parametros = gmodel.param_names
    
    texto =  i[2] +';' 
    for i in parametros:
        coef = round(result.values[i],3)
        texto = texto+ ' ' + i + ' = ' + str(coef)
        


    plt.plot(x, result.best_fit, '-', label=texto + f'\n$R^2$ = ' + str(round(R2,3)))
    
            #texto = texto + f'\n$R^2$ = ' + str(round(R2,3)
    
plt.plot(x, y, '-.k',label = 'Datos experimentales')
plt.ylabel(r'Esfuerzo de Cauchy $\sigma$[MPa]')
plt.xlabel(r'Alargamiento  $\epsilon [\frac{m}{m}]$')
plt.legend()
plt.title('Datos de Control')
plt.grid()
plt.show()


##########################################################################################################################
######################### Caracterizacion Hipoxia ########################################################################

x  , y = landa_hipoxia,sigma_hipoxia



for i in Modelos:
    gmodel = Model(i[0])
    
    if i[1] == 1:
        
        result = gmodel.fit(y, landa=x,c1 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2= 1 - rss/tss
        #print(f"R² = {1 - rss/tss:.3f}")
        
    elif i[1] == 2:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
        
    elif i[1] == 3:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
    

    elif i[1] == 4:
        result = gmodel.fit(y, landa=x,c1 = 1, c2 = 1,c3 = 1)
        rss = (result.residual**2).sum()
        tss = sum(np.power(y - np.mean(y), 2))
        R2 = 1 - rss/tss
        
    parametros = gmodel.param_names
    
    texto =  i[2] +';' 
    for i in parametros:
        coef = round(result.values[i],3)
        texto = texto+ ' ' + i + ' = ' + str(coef)

    plt.plot(x, result.best_fit, '-', label=texto + f'\n$R^2$ = ' + str(round(R2,3)))
    
    
    
plt.plot(x, y, '-.k',label = 'Datos experimentales')
plt.ylabel(r'Esfuerzo de Cauchy $\sigma$[MPa]')
plt.xlabel(r'Alargamiento  $\epsilon [\frac{m}{m}]$')
plt.title('Datos Hipoxia')
plt.legend()
plt.grid()
plt.show()

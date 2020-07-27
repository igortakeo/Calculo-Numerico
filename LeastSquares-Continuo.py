import numpy as np
import matplotlib.pyplot as plt

def decompLU(A):
	L = np.eye(np.shape(A)[0]) #matriz L com a dimensao de A com a diagonal principla igual a 1
	U = np.zeros(np.shape(A)) #matriz U full zero
	lim = np.shape(A)[1] #pegando o tamanho da coluna de A (matriz quadrada)	
	sum = 0	

	for i in range(0, lim):
		for j in range(0, lim):
			sum = 0
			#Parte do somatorio de U que vai de 0 a i-1 e acumula L[i][k]*U[k][j] sendo k a variavel iteradora
			for k in range (0, i):
				sum += L[i,k]*U[k,j]
			U[i,j] = A[i,j] - sum #atribuindo o valor de U[i][j]
	
		for j in range (0, lim):
			sum = 0
			#Parte do somatorio de L que vai de 0 a i-1 e acumula L[j][k]*U[k][i] sendo k a variavel iteradora
			for k in range (0, i):
				sum+= L[j,k]*U[k,i]
			L[j,i] = (A[j,i] - sum)/U[i,i] #atribuindo o valor L[i][j]
	
	return L, U #retornando o resultado

def solveLU(L, U, b):
	#Resolvendo o sistema A*x = b
	#Como A = L*U, entao (L*U)*x = b
	#Vamos fazer L*(U*x) = b e U*x=y, portanto L*y=b
	#Logo, para encontrarmos o resultado resolvemos o sistema U*x = y 

	y = np.zeros(np.shape(b)) #matriz coluna y com a mesma dimensao de b 
	x = np.zeros(np.shape(b)) #matriz coluna x (resultado do sistema) com a mesma dimensao de b		
	
	for i in range (0, np.shape(b)[0]): #resolvendo o sistema  L*y=b
		for j in range (0,i):
			y[i] -= y[j]*L[i,j]
		y[i]+=b[i]					

	for i in range (np.shape(b)[0]-1, -1, -1): #resolvendo o sistema U*x=b
		for j in range (np.shape(b)[0]-1, i, -1):
			x[i] -= x[j]*U[i,j]
		x[i]+=y[i]
		x[i]/=U[i,i]

	return x #retornando o resultado



A = np.array([[1.0, 1.0/2.0, 1.0/3.0],  
              [1.0/2.0, 1.0/3.0, 1.0/4.0],
              [1.0/3.0, 1.0/4.0, 1.0/5.0]], dtype='double')				

e = np.exp(1);
b = [e-1,1,e-2];

(L, U) = decompLU(A)
alpha = solveLU(L, U, b)

print('Matriz A:\n{}\n'.format(A))
print('Matriz B:\n{}\n'.format(b))
print('Matriz X:\n{}\n'.format(alpha))

p = lambda x: alpha[0] + alpha[1]*x + alpha[2]*x**2;
x = np.linspace(0, 1, num=41, endpoint=True)

# Vamos plotar os resultados
plt.figure(figsize=(10,6),facecolor='white')
plt.plot(x,np.exp(x),label = 'f(x)',linewidth = 3)
plt.plot(x,p(x),label = 'p(x)',linewidth = 2,marker='>')
plt.xlabel('x',fontsize='large') 
plt.ylabel('y',fontsize='large') 
plt.title('Comparação da função aproximada') 
plt.legend(fontsize='large') 
plt.show()


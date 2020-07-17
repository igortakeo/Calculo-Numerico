import numpy as np
import sympy

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


A = np.array([[1, 2,0], [1, 3,1],[-2, 0,1]])
b = np.array([3,5,-1])
(L, U) = decompLU(A)
x = solveLU(L, U, b)
print('Matriz L:\n{}'.format(L))
print('Matriz U:\n{}'.format(U))
print('Solve LU:{}'.format(x))

A = sympy.Matrix([[1, 2,0], [1, 3,1],[-2, 0,1]])
b = sympy.Matrix([3, 5,-1])
L, U, _ = A.LUdecomposition()
x = A.solve(b)

print('Matriz L funcao do python:\n{}'.format(L))
print('Matriz U funcao do python:\n{}'.format(U))
print('Solve LU: {}'.format(x))

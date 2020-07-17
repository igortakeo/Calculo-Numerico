import numpy as np

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


def inverse_method(A, error_tol):
	lim = 10000 #definindo um limite de iteracoes
	error = np.inf #atribuindo infinito para o erro
	n = A.shape[1] # pegando a dimensao da matriz A quadrada
	y0 = np.zeros(n)
	y0[0] = 1 #chute inicial normalizado 	
	
	(L, U) = decompLU(A) #fazendo a decomposicao LU de A

	for k in range (0, lim):
		xk = solveLU(L, U, y0) #resolvendo o sistema Ax^k = y^(k-1) com decomposicao LU
		yk = xk/np.linalg.norm(xk) #normalizando xk
		error = np.abs(np.abs(y0.dot(yk))-1) #teste de alinhamento			
		if error <= error_tol: #se o erro for menor que a tolerancia recebida como parametro, pode parar a iteracao
			break
		y0 = yk
	
	lambda_1 = y0.dot(A.dot(y0)) #calculando o lambda_1 = y^k * (A * y^k)
	
	return lambda_1, y0 #retornando o autovalor e seu autovetor associado			




A = np.array([[12, 2, 3],  
              [ 2, 3, 5],
              [ 3, 5,-2]], dtype='double')

(D, V) = np.linalg.eig(A) #usando uma funcao pronta para calcular todos os autovalores
print('Método do Python: {0:.9f}'.format(np.min(abs(D))))
(autovalor, autovetor) = inverse_method(A,0.000000001) #usando o metodo da potencia inversa implementado com erro em torno de 10^-9
print('Método Implementado: {0:.9f}'.format(autovalor))  


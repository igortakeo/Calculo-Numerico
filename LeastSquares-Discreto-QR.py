import numpy as np

def QRDecomposition(A):
	n = np.shape(A)[0] #pegando o tamanho das linhas de A
	m = np.shape(A)[1] #pegando o tamanho das colunas de A
	Q = np.zeros((n,m)) #declarando a matriz Q
	R = np.zeros((m,m)) #declarando a matriz R
	
	for j in range(0, m):
		A_column = A[:, j] #pegando as colunas da matriz A
		V = np.zeros(n) #declarando o vetor V
		V = A_column #V igual a coluna j de A
		for i in range (0, j):
			R[i,j] = Q[:,i].dot(A_column) #fazendo o calculo do R[i,j] = coluna i de Q * coluna j de A ( i != j)
			V -= (Q[:,i].dot(A_column))*Q[:,i] #fazendo o calculo de V 			
		R[j,j] = np.linalg.norm(V) # R[j,j] = norma da coluna j de A (i == j)	
		Q[:,j] = V/np.linalg.norm(V) #normalizando V e atribuindo a coluna j de Q
		
	return Q, R			

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



def LeastSquaresQR(x, y, n):
	#Realizar a resolucao do sistema R*alpha = (Q^T)*y
	V = np.vander(x,n) #criando a matriz de vandermonde V
	(Q,R) = QRDecomposition(V) #fazendo a decomposicao V = Q*R
	b = np.transpose(Q).dot(y) #criando a matriz b = (Q^T)*y
	(L,U) = decompLU(R) #fazendo a decomposicao LU de R
	alpha = solveLU(L,U,b) #resolvendo o sistema usando a decomposicao LU (R*alpha = b / L*U*alpha = b)
	return alpha[::-1] #retornando o vetor ao contrario		


x = [-1.0, 0.0, 1.0, 2.0];
y = [0.0, -1.0, 0.0, 7.0];
print(x)
print(y)

alpha = LeastSquaresQR(x, y, 2+1) #como queremos uma aproximacao para um polinomio do segundo grau o n = 2+1 

print(alpha)

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


def FrancisMethod(A, error):

	m = np.inf #declarando m como infinito
	Autovetores = np.eye(np.shape(A)[0]) #declarando a matriz de autovetores como uma matriz indentidade	
	while m > error:
		(Q, R) = QRDecomposition(A) #fazendo a decomposicao QR de A
		A = R.dot(Q) #fazendo A = R*Q
		Autovetores = Autovetores.dot(Q) #pegando os autovetores V = Q1 * Q2 * Q3 * ... 
		m = -np.inf #declarando m como menos infinito
		#pegando o maior valor da matriz A fora da diagonal principal
		for i in range (0, np.shape(A)[0]):
			for j in range (0, np.shape(A)[1]):
				if (i != j) and (m < np.abs(A[i,j])):
					m = np.abs(A[i,j])
	
	
	Autovalores = np.diag(A) #pegando a diagonal principal da matriz A
	
	return Autovalores, Autovetores #retornando o resultado

'''	
A = np.array([[2,0,1], [0,1,0], [1,0,1]], dtype = 'double')
(Aut, Vet) = FrancisMethod(A, 0.01)
print(Aut)
'''
A = np.array([[2,1,1],[1,3,5], [1,5,14]], dtype = 'double')
(Aut, Vet) = FrancisMethod(A, 0.000001)
print('{}\n\n{}'.format(Aut, Vet)) 


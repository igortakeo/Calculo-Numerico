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


def QRDecompositionModificada(A):
	n = np.shape(A)[0] #pegando o tamanho das linhas de A
	m = np.shape(A)[1] #pegando o tamanho das colunas de A
	Q = np.zeros((n,m)) #declarando a matriz Q
	R = np.zeros((m,m)) #declarando a matriz R
	V = np.copy(A) #copiando A para V

	for j in range(0, m):
		for i in range (0, j):
			R[i,j] = Q[:,i].dot(V[:,j]) #fazendo o calculo do R[i,j] = coluna i de Q * coluna j de V ( i != j)
			V[:,j] -= (Q[:,i].dot(V[:,j]))*Q[:,i] #fazendo o calculo de V 			
		R[j,j] = np.linalg.norm(V[:,j]) # R[j,j] = norma da coluna j de V (i == j)	
		Q[:,j] = V[:,j]/np.linalg.norm(V[:,j]) #normalizando V e atribuindo a coluna j de Q
		
	return Q, R			




A = np.array([[1,2],[1,3],[-2,0]], dtype='double')
B = np.array([[3,1], [4,-1]], dtype = 'double')

print('Decomposicao QR classica\n')

(Q, R) = QRDecomposition(A)
print('{}\n\n{}\n\n{}'.format(Q, R, Q.dot(R)))
print('\n')
(Q, R) = QRDecomposition(B)
print('{}\n\n{}\n\n{}'.format(Q, R, Q.dot(R)))

print('\n\n')

A = np.array([[1,2],[1,3],[-2,0]], dtype='double')
B = np.array([[3,1], [4,-1]], dtype = 'double')

print('Decomposicao QR do Python\n')

(Q_python,R_python) = np.linalg.qr(A)
print('{}\n\n{}\n\n{}'.format(Q_python, R_python, Q_python.dot(R_python)))
print('\n')
(Q_python,R_python) = np.linalg.qr(B)
print('{}\n\n{}\n\n{}'.format(Q_python, R_python, Q_python.dot(R_python)))

print('\n\n')

A = np.array([[1,2],[1,3],[-2,0]], dtype='double')
B = np.array([[3,1], [4,-1]], dtype = 'double')

print('Decomposicao QR modificada\n')

(Q, R) = QRDecompositionModificada(A)
print('{}\n\n{}'.format(Q, R))
print('\n')
(Q, R) = QRDecomposition(B)
print('{}\n\n{}'.format(Q, R))




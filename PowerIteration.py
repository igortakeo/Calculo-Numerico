import numpy as np

def power_method(A, error_tol):
	lim = 10000 #Numero maximo de iteracoes
	error = np.inf #atribuicao de infinito para o erro
	n = A.shape[1] #pegando a dimensao da matriz A quadrada
	y0 = np.zeros(n)  
	y0[0] = 1 #chute inicial normalizado

	for k in range (0, lim):	
		xk = A.dot(y0) #produto de matrizes (x^k) = A * y^(k-1) 
		yk = xk/np.linalg.norm(xk) #normalizando xk	
		error = np.abs(np.abs(y0.dot(yk))-1) #teste de alinhamento, calculando o erro
		if error <= error_tol: # se o erro for menor que a tolerancia recebida como parametro, pode para a iteracao
			break
		y0 = yk #atribuindo o novo y0
 		

	lambda_1 = y0.dot(A.dot(y0)) #calculando o lambda_1 = y^k * (A * y^k)
	
	return lambda_1, y0 #retornando o autovalor e seu autovetor associado


A = np.array([[12, 2, 3],  
              [ 2, 3, 5],
              [ 3, 5,-2]], dtype='double')

(D_python,V_python) = np.linalg.eig(A); # usando uma função pronta para calcular todos os autovalores
print('Método do Python: {0:.15f}'.format(np.max(abs(D_python))))
(autovalor, autovetor) = power_method(A, 0.000000001) # usando o metodo da potencia implementado com erro em torno de 10^-9
print('Método Implementado: {0:.15f}'.format(autovalor))

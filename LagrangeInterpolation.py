import numpy as np
import matplotlib.pyplot as plt

def LagrangeInterpolation(xi, yi, points):
	m = points.size
	l = np.ones(xi.shape, dtype='double')
	ans = np.zeros(m)
	
	for j in range (0, m):	
		for k in range(0, len(xi)): 
			for i in range(0, len(xi)):
				if i != k:
					l[k] *= (points[j]-xi[i])/(xi[k]-xi[i])	 
		ans[j] = yi.dot(l)
		l = np.ones(xi.shape)
		
	return ans

		
values_x = np.array([-2,0,3,5], dtype='double')
values_y = np.array([3,-2,4,2], dtype='double')
points_x = np.linspace(-2, 5, num=41, endpoint=True)
points_y = LagrangeInterpolation(values_x, values_y, points_x)

plt.figure(figsize=(10,6),facecolor='white')
plt.plot(points_x,points_y,label = 'Interpolação de Lagrange',linewidth = 2)
plt.scatter(values_x,values_y,label = 'Pontos',linewidth = 2) 
plt.xlabel('x',fontsize='large') 
plt.ylabel('y',fontsize='large') 
plt.title('') 
plt.legend() 
plt.show()


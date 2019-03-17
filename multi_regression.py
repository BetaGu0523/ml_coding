#-*-cding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt

def regression(x_data, y_data, alpha=0.006):
	# 获取样本数量与特征数量 
	m = x_data.shape[0]
	n = x_data.shape[1]

	# 初始化theta的值
	theta = np.zeros(n)

	loop = 0

	cost_sum = []

	# 梯度下降
	while loop<1000:
		theta_temp = theta.copy() # 容易出错

		for j in range(n):
			sum = 0
			for i in range(m):
				sum = sum + (np.dot(theta,x_data[i])-y_data[i])*x_data[i,j]
			theta_temp[j] = theta_temp[j] - sum * alpha
		
		theta = theta_temp  # 容易出错
		cost_sum.append(np.dot((np.dot(x_data,theta.T)-y_data).T,np.dot(x_data,theta.T)-y_data))
		loop += 1

	return np.array(cost_sum), theta_temp


def main():
	# 构建x_0列
	ones_column = np.ones((100,1))

	# 构建x
	x = np.tile(np.arange(10),100)
	np.random.shuffle(x)
	x = x.reshape((100,10))
	x = x/10
	# 组合x_data
	x_data = np.hstack((ones_column,x))

	#预设theta
	theta = np.arange(11) 

	# 构建y_data
	y_data = np.dot(x_data,theta) + np.random.randn(100)

	# 回归返回theta与代价函数值
	cost_sum ,eli_theta = regression(x_data,y_data)

	#plt.plot(cost_sum,np.arange(cost_sum.shape[0]))
	#plt.show()

	eli_theta_2 = np.dot(np.linalg.inv(np.dot(x_data.T,x_data)),x_data.T)@y_data

	print (eli_theta)
	print (eli_theta_2)




if __name__ == '__main__':
	main()
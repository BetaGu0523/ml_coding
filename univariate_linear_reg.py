#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt

# 构造一个一元回归类
class nui_linear_reg():
	def __init__(self, x_data, y_data, alpha=0.01, theta_0=0, theta_1=0):
		self.x_data = x_data
		self.y_data = y_data
		self.alpha = alpha
		self.theta_0 = theta_0
		self.theta_1 = theta_1

	# 计算梯度函数
	def d_theta(self):
		d_0_sum = 0
		d_1_sum = 0
		for i in range(self.x_data.size):
			d_0_sum = d_0_sum + (self.theta_0 + self.theta_1*self.x_data[i]-self.y_data[i])
			d_1_sum = d_1_sum + (self.theta_0 + self.theta_1*self.x_data[i]-self.y_data[i])*self.x_data[i]
		return self.theta_0 - self.alpha * d_0_sum/self.x_data.size, self.theta_1 - self.alpha * d_1_sum/self.x_data.size

	def loop(self,loop_num):
		for i in range(loop_num):
			self.theta_0,self.theta_1 = self.d_theta()
		return self.theta_0,self.theta_1
			

def main():
	x = np.arange(10)
	y = 2*x + 1 #+ np.random.random(10)
	
	# 散点图
	plt.scatter(x,y)
	#plt.show()
	#print (x)
	#print (y)

	new_unireg = nui_linear_reg(x,y)
	
	theta_0,theta_1 = new_unireg.loop(5000)
	print (theta_0,theta_1)
	plt.plot(x,theta_0+theta_1*x)
	plt.show()

	
	


if __name__ == '__main__':
	main()
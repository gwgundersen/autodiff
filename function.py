import numpy as np


class Function(object):

	def __init__(self, value, backref):
		self.value = value
		self.backref = backref

	def forward(self):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError
		

class AddFn(Function):

	def forward(self):
		return self.value + self.backref.value

	def backward(self):
		return 1


class MulFn(Function):

	def forward(self):
		return self.value * self.backref.value

	def backward(self):
		return self.value


class ExpFn(Function):

	def forward(self):
		return np.e ** self.backref.value

	def backward(self):
		return np.e ** self.backref.value


class PowFn(Function):

	def forward(self):
		return self.backref.value ** self.value

	def backward(self):
		n = self.value
		x = self.backref.value
		return n * (x ** (n-1))


class LogFn(Function):

	def forward(self):
		return np.log(self.backref.value)

	def backward(self):
		return 1 / float(self.backref.value)


class SinFn(Function):

	def forward(self):
		return np.sin(self.backref.value)

	def backward(self):
		return np.cos(self.backref.value)

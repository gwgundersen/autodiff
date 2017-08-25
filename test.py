from variable import Variable


def f(x):
	a = x.exp()
	b = a**2
	c = a + b
	d = c.exp()
	e = d.sin()
	return d + e


if __name__ == '__main__':
	x = Variable(3)
	out = f(x)
	out.backward()
	print(x.grad)


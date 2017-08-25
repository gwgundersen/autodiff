import function as F


#------------------------------------------------------------------------------

class Variable(object):
	"""`Variable` is a data wrapper for performing automatic differentiation. 
	It uses magic methods, e.g. `__add__`, to create a graph of forward
	operations; when `backward` is called, it computes the gradient of the 
	entire composed function with respect to the original variable.

	Example
	-------
	F(x)	= log(((x + 2)(x + 2))^3)
	F'(x)	= 6 / (x + 2)
	F'(3)	= 1.2

	>>> x = Variable(3)
	>>> y = x + 2
	>>> z = y * y
	>>> w = z**3
	>>> g = w.log()
	>>> g.backward()
	>>> x.grad
	1.2
	"""

	def __init__(self, value, backref=None, CreatorFn=None):
		if not CreatorFn and not backref:
			# When a user creates a `Variable` directly, the instance has no 
			# creator function and its value is just the value passed in.
			self.value = value
			self.backref = None
			self.creator_fn = None
		else:
			# Otherwise, the `Variable` instance is created by operating 
			# against another `Variable` instance, which requires an `Fn` or
			# generating function.
			self.creator_fn = CreatorFn(value, backref)
			self.value = self.creator_fn.forward()
			self.backref = backref
		# `grad` is only set on a call to `backward`. We create it here so the
		# user can check for the property without error.
		self.grad = None

	def backward(self, back_grad=None):
		if back_grad and self.creator_fn:
			# This is the chain rule. We take a previously computed gradient
			# and multiply it by the gradient of the current function.
			self.grad = back_grad * self.creator_fn.backward()
		elif self.creator_fn:
			# This should happen exactly once, for the `Variable` upon which
			# `backward` is called.
			self.grad = self.creator_fn.backward()
		else:
			# This should happen exactly once, for the very first `Variable`.
			self.grad = back_grad

		if self.backref:
			self.backref.backward(self.grad)

#------------------------------------------------------------------------------

	def _op(self, obj, fn):
		backref = self
		if isinstance(obj, self.__class__):
			return Variable(obj.value, backref, fn)
		return Variable(obj, backref, fn)

	def __add__(self, obj):
		return self._op(obj, F.AddFn)

	def __mul__(self, obj):
		if self is obj:
			return self._op(2, F.PowFn)
		return self._op(obj, F.MulFn)

	def __pow__(self, obj):
		return self._op(obj, F.PowFn)

	def log(self, obj):
		return self._op(None, F.LogFn)

	def sin(self):
		return self._op(None, F.SinFn)

	def exp(self):
		return self._op(None, F.ExpFn)

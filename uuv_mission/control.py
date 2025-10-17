class PDController:
	"""
	Simple discrete-time PD controller.
	u[t] = KP * e[t] + KD * (e[t] - e[t-1])
	"""
	def __init__(self, KP: float = 0.15, KD: float = 0.6):
		self.KP = KP
		self.KD = KD
		self.last_error = 0.0

	def reset(self):
		"""Reset internal state between simulations."""
		self.last_error = 0.0

	def step(self, reference: float, observation: float) -> float:
		"""Compute control action given reference and observation (depth)."""
		e = reference - observation
		de = e - self.last_error
		u = self.KP * e + self.KD * de
		self.last_error = e
		return u

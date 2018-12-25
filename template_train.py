class templatetrain(data_loader, config):
	super(templatetrai, self).__init__(sess, model, data, config):
		self.data_loader = data_loader
		self.config = config
		self.model = model 
	
	def train_epoch(self):
		NotImplementedError

	def train_step(self):
		NotImplementedError
	
from base.base_model import BaseModel, BaseModel_IMG
import tensorflow as tf


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        ## Build your model here
        ## 1. Input
        ## 2. Model Architecture
        ## 3. Output
        ## 4. Loss function and Performance(Accuracy )
        ## 5. Optimizer and train
        pass

    def init_saver(self):
#        self.saver = tf.train.Saver(max_to_keep = self.config.max_to_keep,
#                            save_relative_path = True)
        pass

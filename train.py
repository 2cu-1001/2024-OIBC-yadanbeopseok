import models

class OurTrainer:
    def __init__(self):
        self.c2g = models.Climate2Generation()
        self.g2p = models.Generation2Prrice()
        
        self.c2g.max_epoch = 30
        self.g2p.max_epoch = 30


    def train_c2g(self):
        self.c2g.make_dataframe()
        self.c2g.make_dataloader()
        self.c2g.train_model()
        self.c2g.save_model()


    def train_g2p(self):
        self.g2p.make_dataframe()
        self.g2p.make_dataloader()
        self.g2p.train_model()
        self.g2p.save_model()
        
    
    def predict_c2g(self, prediction_dataloader):
        return self.c2g.prediction(prediction_dataloader)
    
    
    def predict_g2p(self, prediction_dataloader):
        return self.g2p.prediction(prediction_dataloader)

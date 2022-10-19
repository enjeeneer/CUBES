'''Defines the Building class '''

class Building:
    def __init__(self,geometry_data,systems_data):
        self.nStoreys = geometry_data
        self.system = systems_data

    def get_idf(self):
        return (self.nStoreys,self.system)
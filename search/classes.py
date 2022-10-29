
class Node:
    def __init__(self, X, Y, scat, long, lat, connections):
        self._x = X
        self._y = Y
        self._scat = scat
        self._long = long
        self._lat = lat
        self._connections = connections

        
    #temp coordinate system to represent lon/lat
    def get_x(self):
        return self._x
    def get_y(self):
        return self._y
    def get_scat_number(self):
        return self._x
    def get_longitude(self):
        return self._y
    def get_latitude(self):
        return self._x
    def get_connections(self):
        return self._y
    #generate loc from given coordinates in constructor
    def set_loc(self, value):
        self._loc = value
    def get_loc(self):
        return self._loc
    
    #properties used by search algo
    def set_parentNode(self, value):
        self._parentNode = value
    def get_parentNode(self):
        return self._parentNode
    
    def set_tDist(self, value):
        self._tDist = value
    def get_tDist(self):
        return self._tDist
    
    def set_distToGoal(self, value):
        self._distToGoal = value    
    def get_distToGoal(self):
        return self._distToGoal
    
    def set_distToStart(self, value):
        self._distToStart = value  
    def get_distToStart(self):
        return self._distToStart
    
    
    #collection of all interconnecting streets as nodes
class Map:
    def __init__(self, start, end, nodes):
        self._nodes = nodes
        self._start = start
        self._end = end
    
    def get_start(self):
        return self._start
    
    def get_end(self):
        return self._end    
    
    def get_node(self, node):
        #check if node location matches a node in _nodes
        for curNode in self._nodes:
            if curNode.get_loc() == node.get_loc():
                return curNode
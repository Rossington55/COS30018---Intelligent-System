from asyncio.windows_events import NULL
import classes
import main
import math

def distToGoal(node):
    return node.distToGoal

def distance_betweeen(node1, node2):
    long1 = node1.get_longitude()
    long2 = node2.get_longitude()
    lat1 = node1.get_latitude()
    lat2 = node2.get_latitude()
    LAT_corrected = lat1 + 0.00155
    LONG_corrected = long1 + 0.00113
    LAT_corrected2 = lat2 + 0.00155
    LONG_corrected2 = long2 + 0.00113
    R = 6371 #Radius of earth in km
    dLat = deg_to_rad(LAT_corrected2-LAT_corrected)
    dLon = deg_to_rad(LONG_corrected2-LONG_corrected)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg_to_rad(LAT_corrected)) * math.cos(deg_to_rad(LAT_corrected2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c 
    return d

def deg_to_rad(deg):
    return deg*(math.pi/180)

#Calculate the speed along the route while accounting for the flow
def speed_at_flow(flow):
    x = 8 * (math.sqrt(-10*(flow-1000))+100)
    return x/25

#return the time it takes to travel a distance considering the flow along the route
def timeBetween(flow, dist):
    speed = speed_at_flow(flow)
    return dist/speed
    
#find the travel cost between two scats sites
def getCostofScats(node1, node2, time):
    distBetween = distance_betweeen(node1, node2)
    flow = main.findFlowForSearch(node1.get_scat_number(), time)
    node1.set_flow(flow)
    
    costToScat = timeBetween(node1.get_flow(), distBetween)
    return costToScat

#Returns array of string of each scats number from a single route found by search
def getRouteFromNode(node):
    scatsRoute = []
    scatsRoute.append(node.get_scat_number())
    
    while node.get_parentNode() != NULL:
        node = node.get_parentNode()
        scatsRoute.append(node.get_scat_number())
    
    return reversed(scatsRoute)

#Harrison! call this method from your GUI 
def harrisonsMethod(start, end, time):
    routes = []
    nodes = main.initialise_map('data/data1.xls')
    
    sNode = nodes.get(str(start))
    eNode = nodes.get(str(end))

    print('Scat Start: ' + str(sNode.get_scat_number()))
    print('Scat End: ' + str(eNode.get_scat_number()))
    
    for i in range(5):
        routes.append(getRouteFromNode(aStarSearch(sNode, eNode, time, routes)))
                      
    return routes

#Print all found routes in order of scats sites traveled to from start to finish
def printRoutes(routes):
    for route in routes:
        for scats in route:
            print(scats, " -> ", end =" ")
        print()

def returnDistToGoal(node):
    return node.get_distToGoal()
        

#Call this method and pass the nodes with the start scats site and end scats site and current time
def aStarSearch(start, end, time, routes):
    searchedNodes = []
    goalReached = False
    
    openList = [] 
    openList.append(start)
    curNode = ''
    
    while len(openList) > 0:
        curNode = openList.pop()
        curNode.set_distToStart(distance_betweeen(curNode, start))
        
        for i in openList:
            if curNode.tDist > i.tDist:
                curNode = i          
        
        searchedNodes.append(curNode)
        
        if curNode == end:
            goalReached = True
            break
        
        childNodes = []
        #Check each street/scatsSite from current node
        for adjNode in curNode.get_adjNodes():
            adjNode.parentNode = curNode
            adjNode.set_distToGoal(distance_betweeen(adjNode, end))
            adjNode.set_distToStart(distance_betweeen(adjNode, start))
            childNodes.append(adjNode)
    
        childNodes.sort(key=returnDistToGoal)
        
        for childNode in childNodes:
            skip = False
            #add the cost between the start and the current node with the cost between the current node and the next node
            childNode._distToStart = curNode.get_distToStart() + getCostofScats(curNode, childNode, time) 
            childNode.tDist = childNode.get_distToStart() + childNode.get_distToGoal()
            childNode.parentNode = curNode
            
            for cell in openList:
                if childNode.get_scat_number() == cell.get_scat_number():
                    if childNode.tDist >= cell.tDist:
                        skip = True
                
        if skip:
            openList.append(childNode)
        
    #Return the goal node which can be unpacked to return the route in order of scats sites traveled to
    if goalReached:
        print('Goal Reached')
        return curNode
    print('No Goal Reached')
    return

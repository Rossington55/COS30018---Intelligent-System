import classes

def distToGoal(node):
    return node.distToGoal


def aStarSearch(start, end):
    searchedNodes = []
    goalReached = False
    
    openList = [] 
    openList.append(start)
    
    while openList.count > 0:
        curNode = openList.pop()
        
        for i in len(openList):
            if curNode.tDist > openList[i].tDist:
                curNode = openList[i]          
        
        searchedNodes.append(curNode)
        
        if curNode == end:
            goalReached = True
            break
        
        childNodes = []
        #Check each street/scatsSite from current node
        for adjNode in curNode.adjNodes:
            workingNode = map.getNode(adjNode)
            workingNode.parentNode = curNode
            childNodes.append(workingNode)
    
        childNodes.sort(key=distToGoal)
        
        for childNode in childNodes:
            skip = False
            childNode.distToStart = curNode.distToStart #(+ ?) #add distance between intersections/scats sites
            childNode.tDist = childNode.distToStart + childNode.distToGoal
            childNode.parentNode = curNode
            
            for cell in openList:
                if childNode.X == cell.X and childNode.Y == cell.Y:
                    if childNode.tDist >= map.getNode(cell).tDist:
                        skip = True
                
        if skip:
            openList.append(childNode)
        
    if goalReached:
        return searchedNodes
    return


import numpy as np
from numpy import inf

np.random.seed = 7.

ants = 5
cities = 5
times = 1

#d =np.random.randint(1, 50, size=(20, 20))

d = np.array([[0,2,30,9,1],
              [4,0,47,7,7],
              [31,33,0,33,36],
              [20,13,16,0,28],
              [9,36,22,22,0]])

alpha = 0
beta = 0
e = 0.5
q = 0

np.seterr(divide='ignore', invalid='ignore')
visibility = 1/d
visibility[visibility == inf ] = 0    

pheromne = 0.1*np.ones((ants,cities))

route = np.ones((ants, cities+1))

for t in range(times):
    
    route[:,0] = 1          
    
    for i in range(ants):
        
        copy_of_visibility = np.array(visibility)        
        
        for j in range(cities-1):
            
            combine_feature = np.zeros(ants)    
            cummulative_probability = np.zeros(ants)            
            
            current_location = int(route[i,j]-1)       
            
            copy_of_visibility[:,current_location] = 0    
            
            pheromne_feature = np.power(pheromne[current_location,:],beta)   
            visibility_feature = np.power(copy_of_visibility[current_location,:],alpha)  
            
            pheromne_feature = pheromne_feature[:,np.newaxis]                    
            visibility_feature = visibility_feature[:,np.newaxis]                    
            
            combine_feature = np.multiply(pheromne_feature,visibility_feature)     
                        
            total = np.sum(combine_feature)                       
            
            probs = combine_feature/total   
            
            cummulative_probability = np.cumsum(probs)    
           
            r = np.random.random_sample()  
          
            city = np.nonzero(cummulative_probability>r)[0][0]+1      
            
            route[i,j+1] = city              
           
        left = list(set([i for i in range(1,cities+1)])-set(route[i,:-2]))[0]     
        
        route[i,-2] = left                   
       
    optimal_route = np.array(route)              
    
    storing_distance = np.zeros((ants,1))             
    
    for i in range(ants):
        
        s = 0
        
        for j in range(cities-1):
            
            s = s + d[int(optimal_route[i,j])-1,int(optimal_route[i,j+1])-1]   
        
        storing_distance[i] = s                      
       
    min_dist_location = np.argmin(storing_distance)             
    min_dist_cost = storing_distance[min_dist_location]        
    
    best_route = route[min_dist_location,:]              
    pheromne = (1-e)*pheromne                    
    
    for i in range(ants):
        for j in range(cities-1):
            dt = 1/ storing_distance[i]
            pheromne[int(optimal_route[i,j])-1,int(optimal_route[i,j+1])-1] = pheromne[int(optimal_route[i,j])-1,int(optimal_route[i,j+1])-1] + dt   
           
    
print(int(min_dist_cost[0]) + d[int(best_route[-2])-1,0]) 

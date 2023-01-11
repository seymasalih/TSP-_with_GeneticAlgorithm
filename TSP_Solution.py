import numpy as np, random, operator
import matplotlib.pyplot as plt


def create_starting_population(size,Number_of_city):
    '''Method create starting population 
    size= No. of the city
    Number_of_city= Total No. of the city
    '''
    population = []
    
    for i in range(0,size):
        population.append(create_new_member(Number_of_city))
        #size kadar üye populasyona eklendi (1000 üye)
    return population

#------------------------------------------------------------------------------------
"""
def pick_mate(N):
    '''mates are randomaly picked 
    N= no. of city '''
    i=random.randint(0,N)    
    return i"""
#------------------------------------------------------------------------------------
def distance(i,j):
    '''
    Method calculate distance between two cities if coordinates are passed
    i=(x,y) coordinates of first city
    j=(x,y) coordinates of second city
    '''
    #returning distance of city i and j 
    return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
#------------------------------------------------------------------------------------
def score_population(population, CityList):  
    '''
    Score of the whole population is calculated here
    population= 2 dimensional array conating all the routes
    Citylist= List of the city 
    '''
    scores = []
  
    for i in population:
        #print(i)
        scores.append(fitness(i, CityList))
        #print([fitness(i, the_map)])
    return scores
#------------------------------------------------------------------------------------
def fitness(route,CityList):
    '''Individual fitness of the routes is calculated here
    route= 1d array
    CityList = List of the cities
    '''
    #Calculate the fitness and return it.
    score=0
    #N_=len(route)
    for i in range(0,len(route)):
        if i+1<len(route):
            k=int(route[i])
            l=int(route[i+1])
    
            score = score + distance(CityList[k],CityList[l])

        else :
            k=int(route[i])
            l=int(route[0])
            score = score + distance(CityList[k],CityList[l])
        
    return score
#------------------------------------------------------------------------------------
def create_new_member(Number_of_city):
    '''
    creating new member of the population
    '''
    pop=set(np.arange(Number_of_city,dtype=int)) #1'den Number_of_city'e kadar saydırıyor 1,2,3,4...,24
    route=list(random.sample(pop,Number_of_city)) #pop'u random sıralıyor Number_of_city sayısı kadar 17,4,11,...6
    #yani her eleman bir rota demek
    return route
#------------------------------------------------------------------------------------

#PMX
def crossover(a,b):
    '''
    cross over 
    a=route1
    b=route2
    return child
    '''
    child=[]
    childA=[]
    childB=[]
    

    geneA=int(random.random()* len(a))
    geneB=int(random.random()* len(a))

    start_gene=min(geneA,geneB)
    end_gene=max(geneA,geneB)
    
    for i in range(start_gene,end_gene):
        childA.append(a[i])
        
    childB=[item for item in b if item not in childA]
    child=childA+childB
           
    return child
#------------------------------------------------------------------------------------

#Centre Inverse Mutation (CIM)
def mutate(route, probability):
    
    route_one=[]
    route_two=[]
    
    route_one=route[:13]
    route_two=route[13:]
    
    route_one.reverse()
    route_two.reverse()

    mutated_solution=[]
    mutated_solution= route_one + route_two
    
    return mutated_solution
#------------------------------------------------------------------------------------   
    
def selection(popRanked, eliteSize):
    selectionResults=[]
    result=[]
    for i in popRanked:
        result.append(i[0])
    #print("RESULT", result)
    for i in range(0,eliteSize):
        selectionResults.append(result[i])
    
    return selectionResults
#------------------------------------------------------------------------------------
def elitismRoutes(population,City_List):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(population[i],City_List)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
#------------------------------------------------------------------------------------
def breedPopulation(mating_pool):
    children=[]
    for i in range(len(mating_pool)-1):
            children.append(crossover(mating_pool[i],mating_pool[i+1]))
    return children
#------------------------------------------------------------------------------------
def mutatePopulation(children,mutation_rate):
    new_generation=[]

    #new_generation.append(mutate(children[0],mutation_rate))
    
    for i in range(0,10):
        muated_child=mutate(children[i],mutation_rate)
        new_generation.append(muated_child)
    """
    for i in children:
        muated_child=mutate(i,mutation_rate)
        new_generation.append(muated_child)
    """
    return new_generation
#------------------------------------------------------------------------------------
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
#------------------------------------------------------------------------------------
def next_generation(City_List,current_population,mutation_rate,elite_size):
    population_rank=elitismRoutes(current_population,City_List)
    
    #print(f"population rank : {population_rank}")
    #population_rank_size=len(population_rank)
    selection_result=selection(population_rank,elite_size)
    #print(f"selection results {selection_result}")
    #selection_result=selection(population_rank,population_rank_size)
    mating_pool=matingPool(current_population,selection_result)
    #print(f"mating pool {mating_pool}")

    children=breedPopulation(mating_pool)
    #print(f"childern {children}")

    next_generation=mutatePopulation(children,mutation_rate)

    del children[:10]
    children.extend(next_generation)

    #print(f"next_generation {next_generation}")
    return children
#------------------------------------------------------------------------------------

def genetic_algorithm(City_List,size_population=1000,elite_size=75,mutation_Rate=0.01,generation=300,num_selected=500):
    #size_population=1000 ,elite_size=75 , generation =2000
    '''size_population = 1000(default) Size of population
        elite_size = 75 (default) No. of best route to choose
        mutation_Rate = 0.05 (default) probablity of Mutation rate [0,1]
        generation = 2000 (default) No. of generation  
    '''
    

    pop=[]
    progress = []
    
    Number_of_cities=len(City_List)
    
    population=create_starting_population(size_population,Number_of_cities)

    
#ELİTİSM    
    progress.append(sorted(elitismRoutes(population,City_List))[0][1])
    print(f"initial route distance {progress}")
    #print(f"initial route {population[0]}")

    for i in range(0,generation):
        pop = next_generation(City_List,population,mutation_Rate,elite_size)
        population.extend(pop)
        del population[:74]           
        progress.append(sorted(elitismRoutes(pop,City_List))[0][1])
        print("Generation=",i)

   
    rank_=sorted(elitismRoutes(pop,City_List))[0]
    #progress=sorted(progress)
    #rank_=progress[0]
    
    print("RANK_",rank_)
    print(f"Best Route :{population[rank_[0]]} ")
    print(f"best route distance {rank_[1]}")
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return rank_, pop

#------------------------------------------------------------------------------------    

cityList = []

"""for i in range(0,25):
    x=int(random.random() * 200)
    y=int(random.random() * 200)
    cityList.append((x,y))"""
cityList=[(33, 89), (9, 81), (40, 126), (80, 96), (120, 124), (127, 110), (93, 67), (89, 140), (186, 104), (136, 192), (131, 169), (179, 181), (120, 7), (184, 130), (111, 185), (120, 187), (110, 193), (1, 27), (68, 5), (194, 173), (139, 52), (97, 74), (198, 11), (164, 60), (45, 145)]
rank_,pop=genetic_algorithm(City_List=cityList)

"""x_axis=[]
y_axis=[]
for i in cityList:
    x_axis.append(i[0])
    y_axis.append(i[1]) """
x_axis=[33, 9, 40, 80, 120, 127, 93, 89, 186, 136, 131, 179, 120, 184, 111, 120, 110, 1, 68, 194, 139, 97, 198, 164, 45]
y_axis=[89, 81, 126, 96, 124, 110, 67, 140, 104, 192, 169, 181, 7, 130, 185, 187, 193, 27, 5, 173, 52, 74, 11, 60, 145]
#plt.scatter(x_axis,y_axis)
#plt.show()
#------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2) 
fig.set_figheight(5)
fig.set_figwidth(10)
        # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(x_axis, y_axis)             # plot A
ax[1].scatter(x_axis, y_axis)             # plot B
start_node = 0
distanca = 0.


print(cityList)
N=len(cityList)


best_route = pop[rank_[0]]
print(best_route)
wbest_distance= rank_[1]

for i in range(N):

    if i<24:
        a=int(best_route[i])
        b=int(best_route[i+1])
    
        start_pos = cityList[a]
        #print("STR POS",start_pos)
        end_pos = cityList[b]
        #print("END POS",end_pos)
        ax[1].annotate("",
                xy=start_pos, xycoords='data',
                xytext=end_pos, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    else:
        a=int(best_route[i])
        b=int(best_route[0])
        
        start_pos = cityList[a]
        #print("STR POS",start_pos)
        end_pos = cityList[b]
        #print("END POS",end_pos)
        ax[1].annotate("",
                xy=start_pos, xycoords='data',
                xytext=end_pos, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    

textstr = "Elitism Selection \n PMX , Scramble \nTotal length: %.3f" % (rank_[1])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=8,verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
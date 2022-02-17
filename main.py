#Faezeh Movahedi
#Fatemeh Dinani
#Tamrin 3 AI

import random
from re import X
import time
import math

from numpy import partition


""""##################### For genetics ###########################"""
class Coromozom():

    def __init__(self) :
        self.len = 20
        self.genes = []
        self.fitness = None


    def set_fitness(self):
        self.fitness = 1 / z(self.genes)


    def get_fitness(self):
        return self.fitness
    

    def get_genes(self):
        return self.genes
        
        
    def get_gene(self , i):
        return self.genes[i]


    def set_genes(self , genes):
        self.genes = genes
        self.set_fitness()

    
    def set_gene(self , gene):
        self.genes.append(gene)


    def get_len(self):
        return self.len


    def creat(self):

        for count in range(self.len): #count = place
            flag = True
            
            while flag:
                rand_num = random.randint(1 , 20) #rand_num = Branch

                if rand_num not in self.genes:
                    self.genes.append(rand_num)
                    flag = False

        self.set_fitness()
 


    def crossover(self , parent2): #Modified one-point crossover
        parent1_genes = self.genes
        parent2_genes = parent2.get_genes()

        rand_num = random.randint(1 , 18) #point for one_point crossover

        child1 = Coromozom()
        child2 = Coromozom()

        for count in range(0 , rand_num):
            child1.set_gene( parent1_genes[count] )
            child2.set_gene( parent2_genes[count] )

        # print(child1.get_genes())
        # print(child2.get_genes())

        count1 = rand_num
        count2 = rand_num

        flag = True
        while flag:
            flag1 = True
            flag2 = True
            while(flag1):
                if (parent2_genes[count1]) not in (child1.get_genes()):
                    child1.set_gene( parent2_genes[count1] )
                    flag1 = False
                
                if count1 == 19:
                    count1 = 0

                elif count1 == rand_num - 1:
                    flag1 = False
                    flag = False

                else:
                    count1 +=1
            
            while(flag2):
                if (parent1_genes[count2]) not in (child2.get_genes()):
                    child2.set_gene( parent1_genes[count2] )
                    flag2 = False
                
                if count2 == 19:
                    count2 = 0

                elif count2 == rand_num - 1:
                    flag2 = False
                    flag = False

                else:
                    count2 += 1

        
        child1.set_fitness()
        child2.set_fitness()

        return child1 , child2

    
    def mutation(self , Pm , Pg): #Mutatopn with movement genes #Pm : Possibility of chromosome mutation  , Pg = Possibility of genes mutation

        rand_num1 = random.random() #between 0 and 1
        num_mutation_genes = []

        if rand_num1 < Pm:
            # print('yes') #for test

            for num_gene in range(20):
                if num_gene not in num_mutation_genes: #To prevent mutations in duplicate genes
                    rand_num2 = random.random()

                    if rand_num2 < Pg:
                        num_mutation_genes.append(num_gene)

                        flag = True
                        while flag:
                            num_gene2 = random.randint(0 , 19)

                            if num_gene2 not in num_mutation_genes: #To prevent mutations in duplicate genes
                                # print('yes gen = ' , num_gene , 'and gene = ' , num_gene2 , '') #for test
                                temp = self.genes[num_gene]
                                self.genes[num_gene] = self.genes[num_gene2]
                                self.genes[num_gene2] = temp

                                num_mutation_genes.append(num_gene2)

                                flag = False

        self.set_fitness()


    def print(self):
        print (self.get_genes())


with open('table1.txt', 'r') as f:
    table_1 = [[int(num) for num in line.split(',')] for line in f]


with open('table2.txt', 'r') as f:
    table_2 = [[int(num) for num in line.split(',')] for line in f]


def D(i , k):
    if( i > 19 or k > 19 or i < 0 or k < 0):
        return False
    elif i > k:
        return table_1 [i][k]
    else:
        return table_1 [k][i]


def W(j , s):
    if( j > 19 or s > 19 or j < 0 or s < 0):
        return False
    elif j > s:
        return table_2[j][s]
    else:
        return table_2[s][j]


def x(i , j , k , s , jenes):
    if jenes[i] == j and jenes[k] == s:
        return 1
    else:
        return 0

    
def z(matrix): # matrix for genetics is coromozom jenes and for PSO is particle X

    z = 0

    for i in range(20):
        for k in range (20):
            j = matrix[i]
            s = matrix[k]
            z += D(i , k)*W(j , s)*x(i , j , k , s , matrix)
    
    return z


def initialze(size_of_population): #Creat Initial population
    population = []

    for count in range(size_of_population):
        individual = Coromozom()
        individual.creat()
        population.append(individual)
    
    return population


def selection(population , size_of_population): #For select parents with roulette wheel

    sum_fitness = 0
    for individual in population:
        # print('fitness = ' , individual.get_fitness()) #For test
        sum_fitness += individual.get_fitness()
    
    # print ('sum_fitness' , sum_fitness) #For test

    probility = [] # probility = list of cumulative probability
    sum_probility = 0
    for individual in population:
        p = sum_probility + ((individual.get_fitness()) / (sum_fitness)) # p = Cumulative probability
        probility.append(p)
        sum_probility = p
    
    # print ('probility = ' , probility) #For test

    rand_num = random.random() #between 0 and 1
    # print ('ran num = ' , rand_num) #For test
    for count2 in range(size_of_population):
        if rand_num < probility[count2]:
            return population[count2]
    

def genetics(n = 150):
    size_of_population = int(input('Please enter the number of population you want (This number must be a multiple of 4) : '))
    population = initialze(size_of_population)

    # print ('initialze population = ') #For test
    # for individul in population:
    #     individul.print()
    print ('!!!Please wait')
    best_fitness = population[0].get_fitness()
    best_individual = population[0]
    for count in range(n):
        # start_time = time.time()
        new_population = []

        size = math.floor(size_of_population/4)
        
        for count2 in range(size):
            while True:
                parent1 = selection(population , size_of_population)
                if parent1 not in new_population:
                     new_population.append(parent1) #Because 1/4 of population are new_population
                     break
            
            while True: # For select parent2 != parent1
                parent2 = selection(population , size_of_population)
                if parent2 not in new_population:
                    new_population.append(parent2)
                    break    

            child1 , child2 = parent1.crossover(parent2)
            
            child1.mutation(0.4 , 0.3)
            child2.mutation(0.4 , 0.3)

            new_population.append(child1)
            new_population.append(child2)
            
            # print ('new population = ')
            # for individual in new_population: #For test
            #     individual.print()
        
        
        population = new_population
        # print('population = ')
        # for individual in population: #For test
        #     individual.print()

        # print("--- %s seconds ---" % (time.time() - start_time))
    
        for individual in population:
            # print ('individul fitness = ' , individual.get_fitness()) #For test
            if best_fitness < individual.get_fitness():
                best_fitness = individual.get_fitness()
                best_individual = individual
        
        # print ('best fitness = ' , best_fitness , '\nbest individual = ' , )#For test
        # best_individual.print()

    return best_individual , best_fitness


"""######################## For PSO #####################"""

class Particle():
    
    def __init__(self):
        self.len = 20
        self.X = []
        self.pbest = []
        self.pbest_fitness = None
        self.V = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
        self.fitness = None

    def set_fitness(self):
        self.fitness = 1 / z(self.X)


    def get_fitness(self):
        return self.fitness
    

    def set_pbest(self):
        if self.pbest==[]:
            self.pbest = self.X
            self.set_pbest_fitness()

        elif self.fitness > self.pbest_fitness:
            self.pbest = self.X
            self.set_pbest_fitness()


    def get_pbest(self):
        return self.pbest


    def set_pbest_fitness(self):
        self.pbest_fitness = self.fitness #Because this attribute evaluated immediately after the diagnosis of being the best


    def get_pbest_fitness(self):
        return self.pbest_fitness


    def get_V(self):
        return self.V


    def get_X(self):
        return self.X


    def get_pbest(self):
        return self.pbest


    def get_len(self):
        return self.len


    def v(self , w , c1 , c2 , gbest):

        wv = []
        for count in range(self.len):
            wv.append(w * self.V[count])
         
        pbest_xi = []
        for count in range(self.len):
            pbest_xi.append(self.pbest[count] - self.X[count])
        # print('pbest_xi' , pbest_xi)#For test

        gbest_xi = []
        Xgbest = gbest.get_X()
        for count in range(self.len): 
            gbest_xi.append(Xgbest[count] - self.X[count])       
        # print('gbest_xi' , gbest_xi)#For test
        r1 = random.random()
        r2 = random.random()

        for count in range(self.len):
            self.V[count] = int(wv[count] + c1*r1*pbest_xi[count] + c2*r2*gbest_xi[count])
        

    def x(self):

        X_next = []

        for count in range(self.len):
            X_next.append(self.X[count] + self.V[count])

        X_next_corrected =[]
        for x in X_next:
            if (x not in X_next_corrected) and x < 21 and x > 0:
                X_next_corrected.append(x)
            
            elif x > 20:
                x_corrected = 20
                while True:
                    if x_corrected not in X_next_corrected and x_corrected not in X_next:
                        X_next_corrected.append(x_corrected)
                        break
                    else:
                        x_corrected -= 1
            
            elif x < 1:
                x_corrected = 1
                while True:
                    if x_corrected not in X_next_corrected and x_corrected not in X_next:
                        X_next_corrected.append(x_corrected)
                        break
                    
                    else:
                        x_corrected += 1
            
            else:
                x_corrected = 1
                while True:
                    if x_corrected not in X_next and x_corrected not in X_next_corrected:
                        X_next_corrected.append(x_corrected)
                        break
                    else:
                        x_corrected += 1
            
        self.X = X_next_corrected
 

    def creat(self):
        for count in range(self.len): #count = place
            flag = True
            
            while flag:
                rand_num = random.randint(1 , 20) #rand_num = Branch

                if rand_num not in self.X:
                    self.X.append(rand_num)
                    flag = False

        self.set_fitness()


def g_best(population):
    gbest = population[0]
    gbest_fitness = population[0].get_pbest_fitness()
    
    for particle in population:
        # print('pbest particle = ' , particle.get_pbest_fitness()) #For test
        if (particle.get_pbest_fitness()) > (gbest_fitness):
            gbest_fitness = particle.get_pbest_fitness()
            gbest = particle
            gbest.set_fitness()
        # print('gbest = ' , gbest_fitness) #For test
    
    return gbest


def initialze_PSO(size_of_population): #Creat Initial population
    population = []

    for count in range(size_of_population):
        particle = Particle()
        particle.creat()
        particle.set_fitness()
        particle.set_pbest()
        population.append(particle)

    return population


def PSO(n , w , c1 , c2):
    size_of_population = int(input('Please enter the number of population you want : '))
    population = initialze_PSO(size_of_population)
    gbest = g_best(population)
    print ('!!!Please wait')

    counter = 0
    while counter != n:
  
        for particle in population:
            particle.v(w , c1 , c2 , gbest)
            particle.x()
            particle.set_fitness()
            particle.set_pbest()
        
        gbest = g_best(population)
        # print ('gbest = ' , gbest.get_X() , 'in n = ' , counter , 'with fitness = ' , gbest.get_pbest_fitness()) #For test
        counter += 1

        # for particle in population: #For test
        #     print (particle.get_X()) 
    
    return gbest
    

"""######################## For Main and Comparison #####################"""

def main():
    print ('\n####################### part A : Genetics #######################\n')

    start_time_genetics = time.time()
    best_individual , best_fitness_genetics = genetics()
    genes = best_individual.get_genes()
    print('best individual = ')
    best_individual.print()
    print('with fitness = ' , best_fitness_genetics , ' and z = ' , z(genes))
    end_time_genetics = time.time()

    print ('\n####################### part B : PSO #######################\n')
    
    start_time_pso = time.time()
    gbest = PSO(150 , 0.9 , 2 , 1.9)
    best_fitness_pso = gbest.get_pbest_fitness()
    print(
        'gbest = ',
        gbest.get_X(),
        'with fitness = ',
        best_fitness_pso,
        'and z = ',
        z(gbest.get_pbest())
    )
    end_time_pso = time.time()

    print ('\n####################### part C : Comparison #######################\n')
    
    if (best_fitness_pso > best_fitness_genetics):
        print('In this run PSO return the better answer\n')
    
    if (best_fitness_pso < best_fitness_genetics):
        print('In this run genetics return the better answer\n')
    
    pso_time = end_time_pso - start_time_pso
    genetics_time = end_time_genetics - start_time_genetics
    print(
        "Geneitic run in --- %s seconds ---" % (genetics_time),
        "\nPSO run in --- %s seconds ---" % (pso_time)
    )
    if pso_time < genetics_time:
        print("Therefore, in terms of time, PSO is better than Genetics")
    
    else:
        print("Therefore, in terms of time, Genetics is better than PSO")
    
    print ('\n####################### THE END #######################\n')



main()

    



"""To test D and W function"""
# Dij = D (i-1 , j-1):
#     Because the values ​​of the rows and columns of the matrix in Python are one less than the normal state of the tables 
# D11= D(0,0)
# D26=D(1,5)
# D103=D(9,2)
# D201=D(19,0)
# D211=D(20,0)
# D2020=D(19,19)

# print('D11 = ', D11)
# print('D26 = ', D26)
# print('D103 = ', D103)
# print('D201 = ', D201)
# print('D2020 = ', D2020)

#Wij = D (i-1 , j-1):
#     Because the values ​​of the rows and columns of the matrix in Python are one less than the normal state of the tables 
# W11= W(0,0)
# W26=W(1,5)
# W103=W(9,2)
# W201=W(19,0)
# W2020=W(19,19)

# print('w11 = ', W11)
# print('w26 = ', W26)
# print('w103 = ', W103)
# print('w201 = ', W201)
# print('w2020 = ', W2020)

"""To test creat function in coromozom"""
# coromozom1 = Coromozom() 
# coromozom1.creat()
# genes = coromozom1.get_genes()
# print (genes)

"""To test crossover function in coromozom"""
# coromozom1 = Coromozom()
# coromozom1.creat()
# genes1 = coromozom1.get_genes()
# print (genes1)

# coromozom2 = Coromozom()
# coromozom2.creat()
# genes2 = coromozom2.get_genes()
# print (genes2)

# child1 , child2 = coromozom1.crossover(coromozom2)
# print('child 1 : ' , child1.get_genes() , '\n child 2 : ' , child2.get_genes())

"""To test mutation function in coromozom"""
# coromozom1 = Coromozom()
# coromozom1.creat()
# genes1 = coromozom1.get_genes()
# print ('coromozom1 = ' , genes1)

# coromozom1.mutation(0.6 , 0.3)
# mutation_genes1 = coromozom1.get_genes()
# print('mutation cromozom = ' , mutation_genes1)

"""To test z function"""
# coromozom1 = [14, 4, 1, 16, 17, 19, 13, 18, 6, 9, 7, 12, 15, 3, 20, 8, 2, 10, 11, 5]
# print (coromozom1)

# minZ = z(coromozom1)
# print(minZ)

"""To test fitness function in coromozom"""
# coromozom1 = Coromozom()
# coromozom1.creat()
# genes = coromozom1.get_genes()
# print (genes)
# fitness = coromozom1.get_fitness()
# print('fitness = ' , fitness)

"""To test print function in Coromozom"""
# coromozom1 = Coromozom()
# coromozom1.creat()
# coromozom1.print()

"""To test initialze function"""
# population = initialze(10)

# print ('population = ')
# for individual in population:
#     individual.print()

"""To test selection function"""
# population = initialze(5)

# print ('population = ')
# for individual in population:
#     individual.print()

# print('***************')

# select_individual = selection(population , 5)
# select_individual.print()

"""To test genetics function"""
# best_individual , best_fitness = genetics()
# genes = best_individual.get_genes()
# print('best individual = ')
# best_individual.print()
# print('with fitness = ' , best_fitness , ' and z = ' , z(genes))

"""To test creat function in particle"""
# p = Particle() 
# p.creat()
# x = p.get_x()
# print (x)

"""To test pbest function"""
# p = Particle() 
# p.creat()
# p.set_fitness()
# p.set_pbest()
# pbest1 = p.get_pbest()

# print ( 'pbest2 = ' , pbest1)

"""To test pbest_fitness in particle"""
# p = Particle() 
# p.creat()
# p.set_fitness()
# p_fitness = p.get_fitness()
# p.set_pbest()
# pbest_fitness = p.get_pbest_fitness()
# print('p_fitness = ' , p_fitness , 'pbest_fitness = ' , pbest_fitness)

"""To test v function in particle"""
# p1 = Particle() 
# p1.creat()
# p1.set_fitness()
# p1.set_pbest()
# p1x = p1.get_X()
# print ('p1 = ' , p1x)
# p2 = Particle() 
# p2.creat()
# p2.set_fitness()
# p2.set_pbest()
# p2x = p2.get_X()
# print ('p2 = ' , p2x)
# population = [p1 , p2]
# gbest1 = gbest(population)
# print('gbest = ' , gbest1.get_X())

# p1.v(0.9 , 2 , 1.9 , gbest1)
# vp = p1.get_V()
# print(vp)


"""To test x function"""
# p1 = Particle() 
# p1.creat()
# p1.set_fitness()
# p1.set_pbest()
# p1x = p1.get_X()
# print ('p1 = ' , p1x)
# p2 = Particle() 
# p2.creat()
# p2.set_fitness()
# p2.set_pbest()
# p2x = p2.get_X()
# print ('p2 = ' , p2x)
# population = [p1 , p2]
# gbest1 = g_best(population)
# print('gbest = ' , gbest1.get_X())

# p1.v(0.9 , 1.9 , 2 , gbest1)
# vp = p1.get_V()

# p1.x()
# x_next = p1.get_X()
# print('x_next = ' , x_next)

"""To test initialize_PSO function"""
# population = initialze_PSO(20)
# for particle in population:
#     print(particle.get_X())

""" To test gbest function"""
# population = initialze_PSO(20)
# for particle in population:
#     print(particle.get_X())

# gbest = g_best(population)

# print('final gbest = ' , gbest.get_pbest_fitness())

"""To test PSO"""
# gbest = PSO(1 , 0.9 , 2 , 2)
# print(
#     'gbest = ',
#     gbest.get_X(),
#     'with fitness = ',
#     gbest.get_pbest_fitness(),
#     'and z = ',
#     z(gbest.get_pbest())
# )
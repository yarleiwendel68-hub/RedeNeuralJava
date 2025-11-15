import numpy 
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1+ numpy.exp(-x) )

def derivada(x):
    return x * (1-x)

def inicialisar_pesos(num_entrada):
    numpy.random.seed(1)
    return 2 * numpy.random.random((num_entrada, 1)) -1

def treino(entradas, saidas, num_interaçoes, taxa_aprendizado):
    num_amostras, num_entrada = entradas.shape
    pesos =  inicialisar_pesos(num_entrada)
    erros = []

    for iter in range(num_interaçoes):
       
        saida1 = sigmoide(numpy.dot(entradas,pesos))

        erro   = saidas -  saida1

        erro2 = numpy.mean(erro**2)
        erros.append(erro2)

        if iter % 1000 == 0 :
            print(f"erro quadratico na iteraçao {iter}:{erro2}")
            ajustes = erro*derivada(saida1)
            pesos+=taxa_aprendizado * numpy.dot(entradas.T,ajustes)
    
    
    return pesos,erros


def avaliacao(entradas,pessos):
    return sigmoide(numpy.dot(entradas,pessos))


entradas = numpy.array([[0,0,1],
                        [1,0,1],
                        [1,1,1],
                        [0,1,1]])
    
saidas = numpy.array([[0],
                      [1],
                      [1],
                      [0]])

num_interaçoes = 10000
taxa_aprendizado = 0.1


pesos,erros = treino(entradas,saidas,num_interaçoes,taxa_aprendizado)
saida1 = avaliacao(entradas,pesos)

print("Pesos")
print(pesos)
print("\nSaidas camada 1")
print(saida1)

plt.plot(erros)
plt.xlabel("Número de Iterações ")
plt.ylabel("Erro Quadratico Médio ")
plt.title("Erro durante o treinamento")
plt.show()

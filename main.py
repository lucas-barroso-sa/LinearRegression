import matplotlib.pyplot as plt
import numpy as np
import time
from MQO import LinearRegression 

#questão 1
data = np.loadtxt("EMG1.csv" , delimiter = ' ')

X = data[:, :2]
Y_rotulos = data[:,2]

N = X.shape[0]
C = 5  # Número de classes do problema
Y = np.zeros((N, C))

for i in range(N):
    classe_atual = int(Y_rotulos[i]) - 1 
    Y[i, classe_atual] = 1

print(f"Dimensão de X: {X.shape}")
print(f"Dimensão de Y: {Y.shape}")
print(f"Dimensão rotulos: {Y_rotulos.shape}")

#Questão 2

# O parâmetro 'c' recebe os rótulos originais para colorir os pontos automaticamente
# 'cmap' define a paleta de cores e 'alpha' controla a transparência
scatter = plt.scatter(X[:, 0], X[:, 1], c=Y_rotulos, cmap='tab10', alpha=0.6, edgecolors='none')

# Configurando a legenda com os nomes das expressões
nomes_classes = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']
plt.legend(handles=scatter.legend_elements()[0], labels=nomes_classes, title="Categorias")

# Nomeando os eixos de acordo com os músculos monitorados
plt.title("Gráfico de Espalhamento: Sinais EMG por Expressão Facial")
plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")

plt.grid(True, linestyle='--', alpha=1)

#questão 4

valores_q = [1, 2, 3, 4, 5, 6] 
melhor_q = 1
melhor_acuracia = 0
limiar_melhoria = 0.005 # Exige pelo menos 0.5% de ganho para justificar o aumento de q

print("\n--- Seleção Automática do Hiperparâmetro q ---")
for grau in valores_q:
    modelo_poly = LinearRegression(X, Y, fit_intercept=True, solver='OLS', q=grau)
    
    inicio = time.time()
    modelo_poly.fit()
    tempo_estimacao = time.time() - inicio
    
    Y_pred_matrix = modelo_poly.predict(X)
    classes_preditas = np.argmax(Y_pred_matrix, axis=1) + 1
    
    acertos = np.sum(classes_preditas == Y_rotulos)
    acuracia = acertos / N
    
    print(f"Grau q={grau} | Acurácia: {acuracia:.4f} | Tempo: {tempo_estimacao:.4f} segundos")
    
    # Avalia o compromisso: a nova acurácia supera a melhor anterior somada ao limiar exigido?
    if acuracia > melhor_acuracia + limiar_melhoria:
        melhor_acuracia = acuracia
        melhor_q = grau
    else:
        print(f"-> Ganho marginal muito baixo. O aumento do custo computacional não se justifica.")
        break # Interrompe o laço, pois o grau anterior já atingiu o ponto de saturação

print(f"\n[+] Script selecionou automaticamente q={melhor_q} como o melhor compromisso.")

# Questão 5 e 6 - Validação Monte Carlo e Resultados
print("\n--- Resultados Monte Carlo (R=500) ---")

# 1. MQO Tradicional
modelo_tradicional = LinearRegression(X, Y, solver='OLS', q=1)
acc_trad = modelo_tradicional.monte_carlo_validation(X, Y, Y_rotulos, R=500)

print(f"MQO Tradicional - Média: {np.mean(acc_trad):.4f} | Desvio: {np.std(acc_trad):.4f} | Max: {np.max(acc_trad):.4f} | Min: {np.min(acc_trad):.4f}")

# 2. Classificador MQO Regularizado (Teste com um lambda, ex: 0.5)z
modelo_reg = LinearRegression(X, Y, solver='RIDGE', q=1, lamb=0.5)
acc_reg = modelo_reg.monte_carlo_validation(X, Y, Y_rotulos, R=500)

print(f"MQO Regularizado - Média: {np.mean(acc_reg):.4f} | Desvio: {np.std(acc_reg):.4f} | Max: {np.max(acc_reg):.4f} | Min: {np.min(acc_reg):.4f}")

# 3. Classificador MQO Polinomial (Usando o melhor_q definido na etapa 4)
# Supondo que a variável melhor_q contenha o valor ideal
modelo_poly = LinearRegression(X, Y, solver='OLS', q=melhor_q,lamb = 0.1)
acc_poly = modelo_poly.monte_carlo_validation(X, Y, Y_rotulos, R=500)

print(f"MQO Polinomial - Média: {np.mean(acc_poly):.4f} | Desvio: {np.std(acc_poly):.4f} | Max: {np.max(acc_poly):.4f} | Min: {np.min(acc_poly):.4f}")

plt.show()



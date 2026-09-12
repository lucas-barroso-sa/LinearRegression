import matplotlib.pyplot as plt
import numpy as np
from MQO import LinearRegression 

#questão 1
data = np.loadtxt("china_gdp.csv" , delimiter = ',', skiprows = 1)

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# coloca o ano entre 0 e 1 (senão x**q fica gigante)
X = (X - np.min(X)) / (np.max(X) - np.min(X))

N = X.shape[0]

print(f"Dimensão de X: {X.shape}")
print(f"Dimensão de y: {y.shape}")

#Questão 2

# O parâmetro 'alpha' controla a transparência dos pontos
# o eixo x do gráfico fica com o ano original só pra visualização
plt.scatter(data[:, 0], y, alpha=0.6)

# Nomeando os eixos de acordo com as variáveis do problema
plt.title("Gráfico de Espalhamento: PIB da China ao longo dos anos")
plt.xlabel("Ano")
plt.ylabel("PIB (US$)")

plt.grid(True, linestyle='--', alpha=1)

#questão 4

# Questão 4 - Seleção Automática de q para o MQO Polinomial
valores_q = [1, 2, 3, 4, 5, 6] 
melhor_q = 1
melhor_r2 = 0
limiar_melhoria = 0.005 # Exige pelo menos 0.5% de ganho para justificar o aumento de q

print("\n--- Seleção Automática do Hiperparâmetro q ---")
for grau in valores_q:
    modelo_poly = LinearRegression(X, y, fit_intercept=True, solver='OLS', q=grau)
    modelo_poly.fit()
    
    y_pred = modelo_poly.predict(X)
    
    # Cálculo do R2 nos dados completos (avaliação inicial)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Grau q={grau} | R2: {r2:.4f}")
    
    # Avalia o compromisso: o novo R2 supera o melhor anterior somado ao limiar exigido?
    if r2 > melhor_r2 + limiar_melhoria:
        melhor_r2 = r2
        melhor_q = grau
    else:
        print(f"-> Ganho marginal muito baixo. O aumento do custo computacional não se justifica.")
        break # Interrompe o laço, pois o grau anterior já atingiu o ponto de saturação

print(f"\n[+] Script selecionou automaticamente q={melhor_q} como o melhor compromisso.")

# Questão 5 e 6 - Validação Random Subsampling e Resultados
print("\n--- Resultados Random Subsampling (R=500) ---")

# 1. Polinomial (Usando o melhor_q definido na etapa 4)
modelo_poly = LinearRegression(X, y, solver='OLS', q=melhor_q)
mse_poly, r2_poly = modelo_poly.random_subsampling_validation(X, y, R=500)

print(f"Polinomial - MSE Média: {np.mean(mse_poly):.4e} | Desvio: {np.std(mse_poly):.4e} | Max: {np.max(mse_poly):.4e} | Min: {np.min(mse_poly):.4e}")
print(f"Polinomial - R2  Média: {np.mean(r2_poly):.4f} | Desvio: {np.std(r2_poly):.4f} | Max: {np.max(r2_poly):.4f} | Min: {np.min(r2_poly):.4f}")

# 2. MQO Tradicional
modelo_tradicional = LinearRegression(X, y, solver='OLS', q=1)
mse_trad, r2_trad = modelo_tradicional.random_subsampling_validation(X, y, R=500)

print(f"MQO Tradicional - MSE Média: {np.mean(mse_trad):.4e} | Desvio: {np.std(mse_trad):.4e} | Max: {np.max(mse_trad):.4e} | Min: {np.min(mse_trad):.4e}")
print(f"MQO Tradicional - R2  Média: {np.mean(r2_trad):.4f} | Desvio: {np.std(r2_trad):.4f} | Max: {np.max(r2_trad):.4f} | Min: {np.min(r2_trad):.4f}")

# 3. MQO Regularizado (lambda = 0.25)
modelo_reg025 = LinearRegression(X, y, solver='RIDGE', q=1, lamb=0.25)
mse_reg025, r2_reg025 = modelo_reg025.random_subsampling_validation(X, y, R=500)

print(f"MQO Regularizado (0,25) - MSE Média: {np.mean(mse_reg025):.4e} | Desvio: {np.std(mse_reg025):.4e} | Max: {np.max(mse_reg025):.4e} | Min: {np.min(mse_reg025):.4e}")
print(f"MQO Regularizado (0,25) - R2  Média: {np.mean(r2_reg025):.4f} | Desvio: {np.std(r2_reg025):.4f} | Max: {np.max(r2_reg025):.4f} | Min: {np.min(r2_reg025):.4f}")

# 4. MQO Regularizado (lambda = 0.5)
modelo_reg05 = LinearRegression(X, y, solver='RIDGE', q=1, lamb=0.5)
mse_reg05, r2_reg05 = modelo_reg05.random_subsampling_validation(X, y, R=500)

print(f"MQO Regularizado (0,5) - MSE Média: {np.mean(mse_reg05):.4e} | Desvio: {np.std(mse_reg05):.4e} | Max: {np.max(mse_reg05):.4e} | Min: {np.min(mse_reg05):.4e}")
print(f"MQO Regularizado (0,5) - R2  Média: {np.mean(r2_reg05):.4f} | Desvio: {np.std(r2_reg05):.4f} | Max: {np.max(r2_reg05):.4f} | Min: {np.min(r2_reg05):.4f}")

# 5. MQO Regularizado (lambda = 0.75)
modelo_reg075 = LinearRegression(X, y, solver='RIDGE', q=1, lamb=0.75)
mse_reg075, r2_reg075 = modelo_reg075.random_subsampling_validation(X, y, R=500)

print(f"MQO Regularizado (0,75) - MSE Média: {np.mean(mse_reg075):.4e} | Desvio: {np.std(mse_reg075):.4e} | Max: {np.max(mse_reg075):.4e} | Min: {np.min(mse_reg075):.4e}")
print(f"MQO Regularizado (0,75) - R2  Média: {np.mean(r2_reg075):.4f} | Desvio: {np.std(r2_reg075):.4f} | Max: {np.max(r2_reg075):.4f} | Min: {np.min(r2_reg075):.4f}")

# 6. MQO Regularizado (lambda = 1)
modelo_reg1 = LinearRegression(X, y, solver='RIDGE', q=1, lamb=1)
mse_reg1, r2_reg1 = modelo_reg1.random_subsampling_validation(X, y, R=500)

print(f"MQO Regularizado (1) - MSE Média: {np.mean(mse_reg1):.4e} | Desvio: {np.std(mse_reg1):.4e} | Max: {np.max(mse_reg1):.4e} | Min: {np.min(mse_reg1):.4e}")
print(f"MQO Regularizado (1) - R2  Média: {np.mean(r2_reg1):.4f} | Desvio: {np.std(r2_reg1):.4f} | Max: {np.max(r2_reg1):.4f} | Min: {np.min(r2_reg1):.4f}")

plt.tight_layout()
plt.savefig("espalhamento_gdp.png", dpi=150)
plt.show()

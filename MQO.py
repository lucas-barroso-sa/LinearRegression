import numpy as np

class LinearRegression:
    def __init__(self, X_train, Y_train, fit_intercept=True, solver='OLS', q=1, lamb=0.0):
        self.q = q
        self.fit_in = fit_intercept
        self.solver = solver
        self.lamb = lamb  # Parâmetro de regularização armazenado na instância
        
        # 1. Aplicamos a expansão polinomial nos dados originais ANTES do intercepto
        self.X = self._expand_features(X_train)
        self.N, self.p = self.X.shape
        
        # 2. Adicionamos o intercepto (coluna de 1s) após a expansão
        if self.fit_in:
            self.X = np.hstack((
                np.ones((self.N, 1)), self.X
            ))
            
        self.Y = Y_train
        
    def _expand_features(self, X):
        """
        Função auxiliar para elevar as características até o grau q.
        """
        if self.q == 1:
            return X
        
        X_poly = np.copy(X)
        # Para cada grau de 2 até q, calculamos a potência e concatenamos as novas colunas
        for grau in range(2, self.q + 1):
            X_poly = np.hstack((X_poly, X ** grau))
            
        return X_poly
        
    def fit(self):
        # MQO Tradicional (e Polinomial)
        if self.solver == 'OLS':
            self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
            
        # MQO Regularizado (Tikhonov)
        elif self.solver == 'RIDGE':
            dimensoes = self.X.shape[1]
            I = np.eye(dimensoes)
            
            # O intercepto na posição [0,0] não deve ser penalizado pelo lambda
            if self.fit_in:
                I[0, 0] = 0.0
                
            self.beta = np.linalg.pinv(self.X.T @ self.X + self.lamb * I) @ self.X.T @ self.Y
            
    def predict(self, X):
        # 3. Os dados de teste DEVEM sofrer a mesma expansão polinomial que o treino
        X_test_poly = self._expand_features(X)
        N, p = X_test_poly.shape
        
        # 4. Adiciona o intercepto aos dados de teste, se necessário
        if self.fit_in:
            Xt = np.hstack((
                np.ones((N, 1)), X_test_poly
            ))
        else:
            Xt = np.copy(X_test_poly)
            
        # 5. Retorna a predição matricial
        return Xt @ self.beta

    def monte_carlo_validation(self, X_full, Y_full_ohe, Y_full_rotulos, R=500, test_size=0.2):
        """
        Executa a validação por amostragem aleatória (Monte Carlo).
        Retorna uma lista com as acurácias de todas as rodadas.
        """
        acuracias = []
        N_total = X_full.shape[0]
        N_train = int(N_total * (1 - test_size)) 
        
        indices = np.arange(N_total)
        
        for rodada in range(R):
            np.random.shuffle(indices)
            
            idx_train = indices[:N_train]
            idx_test = indices[N_train:]
            
            X_train_mc = X_full[idx_train]
            Y_train_mc = Y_full_ohe[idx_train]
            
            X_test_mc = X_full[idx_test]
            Y_test_rotulos_mc = Y_full_rotulos[idx_test]
            
            # Instancia um NOVO modelo repassando todos os hiperparâmetros
            modelo_mc = LinearRegression(
                X_train=X_train_mc, 
                Y_train=Y_train_mc, 
                fit_intercept=self.fit_in, 
                solver=self.solver, 
                q=self.q,
                lamb=self.lamb
            )
            
            modelo_mc.fit()
            
            Y_pred_mc = modelo_mc.predict(X_test_mc)
            classes_preditas = np.argmax(Y_pred_mc, axis=1) + 1
            
            acertos = np.sum(classes_preditas == Y_test_rotulos_mc)
            acuracia_atual = acertos / len(Y_test_rotulos_mc)
            
            acuracias.append(acuracia_atual)
            
        return acuracias

    def random_subsampling_validation(self, X_full, y_full, R=500, test_size=0.2):
        """
        Executa a validação por amostragem aleatória (Random Subsampling).
        Retorna duas listas com o MSE e o R2 de todas as rodadas.
        """
        mses = []
        r2s = []
        N_total = X_full.shape[0]
        N_train = int(N_total * (1 - test_size)) 
        
        indices = np.arange(N_total)
        
        for rodada in range(R):
            np.random.shuffle(indices)
            
            idx_train = indices[:N_train]
            idx_test = indices[N_train:]
            
            X_train_mc = X_full[idx_train]
            y_train_mc = y_full[idx_train]
            
            X_test_mc = X_full[idx_test]
            y_test_mc = y_full[idx_test]
            
            # Instancia um NOVO modelo repassando todos os hiperparâmetros
            modelo_mc = LinearRegression(
                X_train=X_train_mc, 
                Y_train=y_train_mc, 
                fit_intercept=self.fit_in, 
                solver=self.solver, 
                q=self.q,
                lamb=self.lamb
            )
            
            modelo_mc.fit()
            
            y_pred_mc = modelo_mc.predict(X_test_mc)
            
            # MSE
            erro = y_test_mc - y_pred_mc
            mse_atual = np.mean(erro ** 2)
            
            # R2
            ss_res = np.sum(erro ** 2)
            ss_tot = np.sum((y_test_mc - np.mean(y_test_mc)) ** 2)
            r2_atual = 1 - (ss_res / ss_tot)
            
            mses.append(mse_atual)
            r2s.append(r2_atual)
            
        return mses, r2s

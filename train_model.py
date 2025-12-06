def train_model(train_X, train_y, test_X, modele):
    """entraine un modele ML de classification et retourne des predictions sur
    le dataset de test"""
    modele.fit(train_X,train_y) 
    prediction=modele.predict(test_X)
    return prediction



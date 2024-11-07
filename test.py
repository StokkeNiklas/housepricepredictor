from sklearn.externals import joblib
with open('housing_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
joblib.dump(model, 'housing_price_model_joblib.pkl')

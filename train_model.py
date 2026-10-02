from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES=['voltage','current','power_factor','temperature','load_kw','hour','day_of_week']
TARGET='energy_kwh'

def main():
    if not Path('data/energy_data.csv').exists(): raise FileNotFoundError('Run: python generate_dataset.py')
    df=pd.read_csv('data/energy_data.csv'); X=df[FEATURES]; y=df[TARGET]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model=RandomForestRegressor(n_estimators=250,max_depth=14,random_state=42,n_jobs=-1); model.fit(Xtr,ytr)
    pred=model.predict(Xte); mae=mean_absolute_error(yte,pred); rmse=mean_squared_error(yte,pred)**.5; r2=r2_score(yte,pred)
    af=['voltage','current','power_factor','temperature','load_kw','energy_kwh']
    detector=IsolationForest(n_estimators=200,contamination=.06,random_state=42); detector.fit(df[af])
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    joblib.dump({'model':model,'features':FEATURES},'models/energy_prediction_model.joblib')
    joblib.dump({'model':detector,'features':af},'models/energy_anomaly_model.joblib')
    Path('reports/model_metrics.txt').write_text(f'Smart Energy ML Model Report\n\nTest samples: {len(Xte)}\nMAE: {mae:.4f} kWh\nRMSE: {rmse:.4f} kWh\nR2 Score: {r2:.4f}\n\nSynthetic data; educational use only.\n',encoding='utf-8')
    print(f'MAE={mae:.4f} kWh | RMSE={rmse:.4f} kWh | R2={r2:.4f}')
    print('Saved models in models/')

if __name__=='__main__': main()

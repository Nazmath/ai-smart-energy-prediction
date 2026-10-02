from pathlib import Path
import joblib
import pandas as pd

def main():
    p=Path('models/energy_prediction_model.joblib'); a=Path('models/energy_anomaly_model.joblib')
    if not p.exists() or not a.exists(): raise FileNotFoundError('Run: python generate_dataset.py && python train_model.py')
    pb=joblib.load(p); ab=joblib.load(a)
    sample=pd.DataFrame([{'voltage':229.5,'current':14.2,'power_factor':.93,'temperature':29.0,'load_kw':3.05,'hour':19,'day_of_week':2}])
    energy=pb['model'].predict(sample)[0]; row=sample.copy(); row['energy_kwh']=energy
    result=ab['model'].predict(row[ab['features']])[0]
    print(f'Predicted next-hour energy: {energy:.2f} kWh'); print('Energy condition:', 'ANOMALY' if result==-1 else 'NORMAL')

if __name__=='__main__': main()

from pathlib import Path
import numpy as np
import pandas as pd

ROWS=5000

def main():
    rng=np.random.default_rng(42)
    ts=pd.date_range("2026-01-01", periods=ROWS, freq="h")
    hour=ts.hour.to_numpy(); dow=ts.dayofweek.to_numpy()
    morning=1.8*np.exp(-((hour-8)**2)/8); evening=2.8*np.exp(-((hour-19)**2)/12)
    load=1.2+0.35*np.sin(hour/24*2*np.pi)+morning+evening+np.where(dow>=5,.45,0)+rng.normal(0,.22,ROWS)
    load=np.clip(load,.5,8.5)
    anomaly=rng.random(ROWS)<.06
    load[anomaly]*=rng.uniform(1.35,1.9,anomaly.sum()); load=np.clip(load,.5,12)
    voltage=np.clip(rng.normal(230,3,ROWS),215,245)
    pf=np.clip(rng.normal(.92,.035,ROWS),.75,.99)
    temp=np.clip(26+4*np.sin((hour-6)/24*2*np.pi)+rng.normal(0,1.8,ROWS),18,38)
    current=np.clip(load*1000/(voltage*pf)+rng.normal(0,.25,ROWS),2,55)
    energy=np.clip(load+rng.normal(0,.08,ROWS),.2,12)
    df=pd.DataFrame({'timestamp':ts,'voltage':np.round(voltage,2),'current':np.round(current,2),'power_factor':np.round(pf,3),'temperature':np.round(temp,2),'load_kw':np.round(load,3),'hour':hour,'day_of_week':dow,'energy_kwh':np.round(energy,3),'anomaly':anomaly.astype(int)})
    Path('data').mkdir(exist_ok=True); df.to_csv('data/energy_data.csv',index=False)
    print(f'Generated {len(df)} rows -> data/energy_data.csv')

if __name__=='__main__': main()

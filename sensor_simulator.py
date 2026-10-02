import time
import numpy as np

def generate_reading(rng):
    hour=np.random.randint(0,24); anomaly=rng.random()<.12
    load=1.5+1.5*np.exp(-((hour-8)**2)/10)+2.4*np.exp(-((hour-19)**2)/14)+rng.normal(0,.2)
    if anomaly: load*=rng.uniform(1.4,1.8)
    voltage=np.clip(rng.normal(230,3),215,245); pf=np.clip(rng.normal(.92,.035),.75,.99)
    temp=np.clip(rng.normal(29,2.5),18,40); current=np.clip(load*1000/(voltage*pf),2,55); energy=max(.2,load+rng.normal(0,.08))
    return voltage,current,pf,temp,load,energy,anomaly

def main():
    rng=np.random.default_rng(42); print('Live Smart Energy Sensor Simulator - Ctrl+C to stop\n')
    try:
        while True:
            v,c,pf,t,l,e,a=generate_reading(rng); print(f'Voltage: {v:.1f} V | Current: {c:.2f} A | Load: {l:.2f} kW | Energy: {e:.2f} kWh | Condition: {"ANOMALY" if a else "NORMAL"}'); time.sleep(2)
    except KeyboardInterrupt: print('\nSimulator stopped.')

if __name__=='__main__': main()

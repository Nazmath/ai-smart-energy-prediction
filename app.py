from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='AI Smart Energy Monitor',page_icon='⚡',layout='wide')
P=Path('models/energy_prediction_model.joblib'); A=Path('models/energy_anomaly_model.joblib')
@st.cache_resource
def load():
    if not P.exists() or not A.exists(): return None,None
    return joblib.load(P),joblib.load(A)

def reading(rng,prob):
    hour=int(pd.Timestamp.now().hour); anomaly=rng.random()<prob
    load=1.5+1.5*np.exp(-((hour-8)**2)/10)+2.4*np.exp(-((hour-19)**2)/14)+rng.normal(0,.2)
    if anomaly: load*=rng.uniform(1.4,1.8)
    voltage=np.clip(rng.normal(230,3),215,245); pf=np.clip(rng.normal(.92,.035),.75,.99); temp=np.clip(rng.normal(29,2.5),18,40); current=np.clip(load*1000/(voltage*pf),2,55); energy=max(.2,load+rng.normal(0,.08))
    return {'timestamp':pd.Timestamp.now(),'voltage':voltage,'current':current,'power_factor':pf,'temperature':temp,'load_kw':load,'hour':hour,'day_of_week':int(pd.Timestamp.now().dayofweek),'energy_kwh':energy}

pb,ab=load(); st.title('⚡ AI-Based Smart Energy Consumption Prediction'); st.caption('Laptop-only AI + IoT simulation for educational and portfolio use.')
if pb is None: st.warning('Models missing. Run python generate_dataset.py then python train_model.py'); st.stop()
auto=st.sidebar.checkbox('Auto refresh',True); prob=st.sidebar.slider('Simulated anomaly probability',0.0,.40,.12,.01)
if 'history' not in st.session_state: st.session_state.history=[]
rng=np.random.default_rng(); r=reading(rng,prob); X=pd.DataFrame([{k:r[k] for k in pb['features']}]); energy=pb['model'].predict(X)[0]; ar=X.copy(); ar['energy_kwh']=energy; result=ab['model'].predict(ar[ab['features']])[0]; r['predicted_energy_kwh']=energy; r['status']='⚠️ ANOMALY DETECTED' if result==-1 else '✅ NORMAL'; st.session_state.history.append(r); st.session_state.history=st.session_state.history[-30:]
h=pd.DataFrame(st.session_state.history); c1,c2,c3,c4=st.columns(4); c1.metric('Load',f"{r['load_kw']:.2f} kW"); c2.metric('Predicted Energy',f"{energy:.2f} kWh"); c3.metric('Current',f"{r['current']:.2f} A"); c4.metric('Power Factor',f"{r['power_factor']:.2f}")
st.subheader(f"Current Status: {r['status']}")
for title,col,y,label in [('Live Energy Load','load_kw','kW','Load'),('Predicted Next-Hour Energy','predicted_energy_kwh','kWh','Predicted Energy'),('Current Monitoring','current','A','Current'),('Temperature Monitoring','temperature','°C','Temperature')]:
    fig=go.Figure(); fig.add_trace(go.Scatter(x=h['timestamp'],y=h[col],mode='lines+markers',name=label)); fig.update_layout(title=title,xaxis_title='Time',yaxis_title=y); st.plotly_chart(fig,use_container_width=True)
st.subheader('Recent Sensor Readings'); st.dataframe(h[['timestamp','voltage','current','power_factor','temperature','load_kw','predicted_energy_kwh','status']].tail(10).iloc[::-1],use_container_width=True,hide_index=True)
st.info('Educational note: synthetic sensor data only; not a validated energy-management or electrical-safety system.')
if auto: time.sleep(2); st.rerun()

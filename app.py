import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Beam Shear Lab", layout="wide")
st.title("⚡ Beam Shear Lab")
st.caption("Interactive shear stress and bending stress analysis")

SECTIONS = {
    "Rectangular": [("b width", 100, 10, 400), ("h depth", 200, 20, 600)],
    "Circular": [("D diameter", 100, 10, 500)],
    "I-section": [("bf flange width", 100, 20, 400), ("h total depth", 200, 40, 600),
                  ("tf flange thickness", 10, 2, 60), ("tw web thickness", 6, 2, 40)],
}

def analyse(kind, p, V_kN, M_kNm, n=401):
    V, M = V_kN * 1e3, M_kNm * 1e6
    if kind == "Rectangular":
        b, h = p
        y = np.linspace(-h/2, h/2, n); w = np.full(n, b); I = b*h**3/12
        Q = b/2*(h**2/4-y**2); H=h
    elif kind == "Circular":
        R = p[0]/2; y=np.linspace(-R,R,n); H=2*R
        w=2*np.sqrt(np.clip(R**2-y**2,0,None)); I=np.pi*R**4/4
        Q=(2/3)*np.clip(R**2-y**2,0,None)**1.5
    else:
        bf,h,tf,tw=p
        tf=min(tf,h/2-1); tw=min(tw,bf-1); hw=h-2*tf
        y=np.linspace(-h/2,h/2,n); H=h
        web=np.abs(y)<=hw/2; w=np.where(web,tw,bf)
        I=(bf*h**3-(bf-tw)*hw**3)/12
        Q=np.where(web,bf*tf*(hw+tf)/2+tw/2*(hw**2/4-y**2),
                   bf/2*(h**2/4-y**2))
    tau=V*Q/(I*np.maximum(w,1e-6))
    sigma=M*y/I
    A=np.sum(w)*(y[1]-y[0])
    check=np.sum(tau*w)*(y[1]-y[0])/V if V else 1.0
    return y,w,tau,sigma,I,A,check,V

kind=st.sidebar.selectbox("Cross-section", list(SECTIONS))
V=st.sidebar.slider("Shear force V (kN)",1,500,50)
M=st.sidebar.slider("Bending moment M (kN·m)",0,300,10)
p=[]
for label, default, lo, hi in SECTIONS[kind]:
    p.append(st.sidebar.slider(f"{label} (mm)",lo,hi,default))
y,w,tau,sigma,I,A,check,Vn=analyse(kind,p,V,M)
tmax=float(np.max(tau)); tavg=Vn/A if A else 0
st.metric("Maximum shear stress",f"{tmax:.3f} MPa")
c1,c2,c3=st.columns(3)
with c1:
    fig,ax=plt.subplots()
    ax.fill_betweenx(y,-w/2,w/2,color="cyan",alpha=.6)
    ax.set_title("Cross-section (schematic)"); ax.set_xlabel("Width (mm)"); ax.set_ylabel("y (mm)")
    ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
with c2:
    fig,ax=plt.subplots()
    ax.plot(tau,y); ax.axvline(tavg,ls="--",label=f"Average = {tavg:.3f} MPa")
    ax.plot([tmax],[y[np.argmax(tau)]],"o",label=f"Peak = {tmax:.3f} MPa")
    ax.set_title("Shear stress τ(y) = VQ/(It)"); ax.set_xlabel("τ (MPa)"); ax.set_ylabel("y (mm)")
    ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
with c3:
    fig,ax=plt.subplots()
    ax.plot(sigma,y); ax.axvline(0,color="black",lw=.8)
    ax.set_title("Bending stress σ(y) = My/I"); ax.set_xlabel("σ (MPa)"); ax.set_ylabel("y (mm)")
    ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
st.write(f"**Second moment of area I:** {I:.4e} mm⁴  |  **Area (numerical):** {A:.2f} mm²  |  **Check ∫τdA/V:** {check:.3f}")
st.info("Assumptions: linear elastic beam theory; dimensions in mm, V in kN, M in kN·m. The cross-section view is schematic.")

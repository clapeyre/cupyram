import numpy as np
from pyram.PyRAM import PyRAM
import os

def generate_baseline():
    print("Setting up baseline simulation...")
    
    # Configuration from TestPyRAM.py
    inputs = dict(
        freq=50,
        zs=50,
        zr=50,
        z_ss=np.array([0, 100, 400]),
        rp_ss=np.array([0, 25000]),
        cw=np.array([[1480, 1530],
                     [1520, 1530],
                     [1530, 1530]]),
        z_sb=np.array([0]),
        rp_sb=np.array([0]),
        cb=np.array([[1700]]),
        rhob=np.array([[1.5]]),
        attn=np.array([[0.5]]),
        rmax=50000,
        dr=500,
        dz=2,
        zmplt=500,
        c0=1600,
        rbzb=np.array([[0, 200],
                       [40000, 400]])
    )

    print("Running PyRAM simulation...")
    pyram = PyRAM(
        inputs['freq'], inputs['zs'], inputs['zr'],
        inputs['z_ss'], inputs['rp_ss'], inputs['cw'],
        inputs['z_sb'], inputs['rp_sb'], inputs['cb'],
        inputs['rhob'], inputs['attn'], inputs['rbzb'],
        rmax=inputs['rmax'], dr=inputs['dr'],
        dz=inputs['dz'], zmplt=inputs['zmplt'],
        c0=inputs['c0']
    )
    
    results = pyram.run()
    print("Simulation complete.")

    output_file = os.path.join(os.path.dirname(__file__), 'baseline_data.npz')
    print(f"Saving baseline data to {output_file}...")
    
    np.savez(
        output_file,
        # Inputs
        freq=inputs['freq'],
        zs=inputs['zs'],
        zr=inputs['zr'],
        z_ss=inputs['z_ss'],
        rp_ss=inputs['rp_ss'],
        cw=inputs['cw'],
        z_sb=inputs['z_sb'],
        rp_sb=inputs['rp_sb'],
        cb=inputs['cb'],
        rhob=inputs['rhob'],
        attn=inputs['attn'],
        rbzb=inputs['rbzb'],
        rmax=inputs['rmax'],
        dr=inputs['dr'],
        dz=inputs['dz'],
        zmplt=inputs['zmplt'],
        c0=inputs['c0'],
        
        # Outputs
        ranges=results['Ranges'],
        depths=results['Depths'],
        tl_grid=results['TL Grid'],
        tl_line=results['TL Line'],
        cp_grid=results['CP Grid'],
        cp_line=results['CP Line'],
        proc_time=results['Proc Time']
    )
    print("Baseline data saved successfully.")

if __name__ == "__main__":
    generate_baseline()

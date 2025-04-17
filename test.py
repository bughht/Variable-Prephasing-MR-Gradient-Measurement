from VP_measure import VP_measure
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=80,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        grad_raster_time=10e-6,
    )

    G = {
        "x": 2e5*(1-np.exp(-1e-2*np.arange(128)))*np.exp(1j*np.linspace(0, 4*np.pi, 128)).real,
        "y": 2e5*(1-np.exp(-1e-2*np.arange(128)))*np.exp(1j*np.linspace(0, 4*np.pi, 128)).imag,
        "z": None
    }

    G["x"] = np.hstack([G["x"], G["x"][::-1]])
    G["y"] = np.hstack([G["y"], G["y"][::-1]])

    dwell = 10e-6
    n_samples = 256

    TR = 100e-3
    TE = 3e-3

    rf_flip_angle = np.deg2rad(30)
    rf_duration = 3e-3
    rf_slice_thickness = 10e-3
    rf_slice_position = np.array([-30e-3, -15e-3, 0, 15e-3, 30e-3])
    VP_steps = 10
    VP_repeat = 5

    vp = VP_measure(G, dwell, n_samples, sys, TR, TE,
                    rf_flip_angle, rf_duration, rf_slice_thickness,
                    rf_slice_position, VP_steps, VP_repeat)
    seq = vp.build_sequence()

    # seq.plot()
    seq.write("test_camera.seq")

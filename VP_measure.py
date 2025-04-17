import numpy as np
import pypulseq as pp
from decimal import Decimal


class VP_measure:

    def __init__(self,
                 # Gradient measurement parameters
                 G: dict,
                 dwell_time: float,
                 n_ADC_samples: int,
                 # System parameters
                 system: pp.Opts,
                 TR: float,
                 TE: float,
                 # RF parameters
                 rf_flip_angle: float,
                 rf_duration: float,
                 rf_slice_thickness: float,
                 rf_slice_position: np.ndarray,
                 # Variable Prephasing parameters
                 VP_steps: int,
                 VP_repeat: int,
                 VP_range=None):
        self.G = G
        self.dwell_time = dwell_time
        self.n_ADC_samples = n_ADC_samples
        self.system = system
        self.TR = TR
        self.TE = TE
        self.rf_flip_angle = rf_flip_angle
        self.rf_duration = rf_duration
        self.rf_slice_thickness = rf_slice_thickness
        self.rf_slice_position = rf_slice_position
        self.VP_steps = VP_steps
        self.VP_repeat = VP_repeat
        self.VP_range = VP_range

        self.K = {
            "x": np.cumsum(G["x"])*self.dwell_time if G["x"] is not None else np.array([0]),
            "y": np.cumsum(G["y"])*self.dwell_time if G["y"] is not None else np.array([0]),
            "z": np.cumsum(G["z"])*self.dwell_time if G["z"] is not None else np.array([0])
        }

        if self.VP_range == None:
            self.VP_range = np.array([min(self.K["x"].min(), self.K["y"].min(), self.K["z"].min()), max(
                self.K["x"].max(), self.K["y"].max(), self.K["z"].max())])

        self.prep_sequence()

    def prep_sequence(self):
        self.rf, self.grad_rf_ss, grad_rf_rephase = pp.make_sinc_pulse(
            flip_angle=self.rf_flip_angle,
            duration=self.rf_duration,
            slice_thickness=self.rf_slice_thickness,
            apodization=0.42,
            time_bw_product=4,
            system=self.system,
            return_gz=True,
            delay=self.system.rf_dead_time,
        )
        self.rf.delay = self.grad_rf_ss.rise_time
        self.grad_rf_ss.delay = 0
        grad_prephase_max = pp.make_trapezoid(
            channel="z", area=np.max(np.abs(self.VP_range+grad_rf_rephase.area)), system=self.system)
        self.grad_prephase_duration = pp.calc_duration(
            grad_prephase_max)
        self.grad_prephase_areas = np.linspace(
            self.VP_range[0], self.VP_range[1], self.VP_steps) + grad_rf_rephase.area

        self.measure_grad = {
            "x": pp.make_extended_trapezoid(
                channel="x",
                amplitudes=self.G["x"],
                times=np.arange(self.G["x"].size) *
                self.system.grad_raster_time,
                system=self.system,
            ) if self.G["x"] is not None else None,
            "y": pp.make_extended_trapezoid(
                channel="y",
                amplitudes=self.G["y"],
                times=np.arange(self.G["y"].size) *
                self.system.grad_raster_time,
                system=self.system,
            ) if self.G["y"] is not None else None,
            "z": pp.make_extended_trapezoid(
                channel="z",
                amplitudes=self.G["z"],
                times=np.arange(self.G["z"].size) *
                self.system.grad_raster_time,
                system=self.system,
            ) if self.G["z"] is not None else None,
        }
        self.measure_grad_duration = max(
            [pp.calc_duration(grad) for grad in self.measure_grad.values() if grad is not None])

        self.adc = pp.make_adc(
            num_samples=self.n_ADC_samples,
            dwell=self.dwell_time,
            system=self.system,
        )
        self.adc.delay = 0

        self.TE_delay = self.TE - \
            pp.calc_duration(self.grad_rf_ss)/2 - \
            self.grad_prephase_duration
        self.TR_delay = self.TR - pp.calc_duration(
            self.grad_rf_ss) - self.grad_prephase_duration - self.TE_delay - self.measure_grad_duration
        if self.TE_delay < 0:
            self.TE_delay = 0
            print("TE_delay < 0, set to 0, TE = {}".format(
                pp.calc_duration(self.grad_rf_ss)/2 + self.grad_prephase_duration))
        if self.TR_delay < 0:
            self.TR_delay = 0
            print("TR_delay < 0, set to 0, TR = {}".format(
                pp.calc_duration(self.grad_rf_ss) + self.grad_prephase_duration + self.TE_delay + self.measure_grad_duration))

    def build_sequence(self) -> pp.Sequence:
        seq = pp.Sequence(self.system)

        for orientation in ["x", "y", "z"]:
            if self.G[orientation] is None:
                continue
            self.grad_rf_ss.channel = orientation
            for slice_pos in self.rf_slice_position:
                self.rf.freq_offset = self.grad_rf_ss.amplitude * slice_pos
                for VP_step_idx in range(self.VP_steps):
                    grad_prephase = pp.make_trapezoid(
                        channel=orientation,
                        area=self.grad_prephase_areas[VP_step_idx],
                        duration=self.grad_prephase_duration,
                        system=self.system,
                    )
                    for VP_repeat_idx in range(self.VP_repeat):
                        seq.add_block(self.rf, self.grad_rf_ss,)
                        seq.add_block(grad_prephase)
                        seq.add_block(pp.make_delay(self.TE_delay))
                        seq.add_block(self.measure_grad[orientation], self.adc)
                        seq.add_block(pp.make_delay(self.TR_delay))

        return seq

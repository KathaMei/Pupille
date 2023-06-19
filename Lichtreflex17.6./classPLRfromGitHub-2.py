#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:22:31 2023

@author: Katharina
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from typing import List


class PLR:
    """Class to handle data representing a pupil response to a flash of light."""

    # TODO: add time stuff
    def __init__(
        self,
        plr: Sequence,
        sample_rate: int,
        onset_idx: int,
        stim_duration: int,
    ) -> None:
        """Initialise the PLR data.
        Parameters
        ----------
        plr : arraylike
            Data representing a pupil response to a flash of light.
        sample_rate : int
            Frequency at which the data were sampled.
        onset_idx : int
            Ordinal index matching the onset of the light stimulus.
        stim_duration : int
            Duration of the light stimlus in seconds.
        Returns
        -------
        None.
        """
        self.plr = plr
        self.sample_rate = sample_rate
        self.onset_idx = onset_idx
        self.stim_duration = stim_duration
    


    def velocity_profile(self) -> np.array:
        """Return the velocity profile of the PLR. Assumes the samples are
        evenly spaced, which is not the case with Pupil Labs data. Smoothing
        and averaging across multiple PLRs should remove cause for concern.
        """
        t = 1 / self.sample_rate
        return np.diff(self.plr, prepend=np.nan) / t

    def acceleration_profile(self) -> np.array:
        """Return the acceleration profile of a PLR. Assumes the samples are
        evenly spaced, which is not the case with Pupil Labs data. Smoothing
        and averaging across multiple PLRs should remove cause for concern.
        """
        t = 1 / self.sample_rate
        vel = self.velocity_profile()
        return np.diff(vel, prepend=np.nan) / t

    def baseline(self) -> float:
        """Return the average pupil size between the start of s and onset_idx."""
        return np.mean(self.plr[0 : self.onset_idx])

    def pupil_size_at_onset(self) -> float:
        """Return pupil size at stimulus onset."""
        return self.plr[self.onset_idx]

    def latency_idx_a(self) -> int:
        """Return the index where pupil size passes 1% change from size at
        light  onset.
        """
        b = self.pupil_size_at_onset()
        threshold = b - (b * 0.01)
        lidx = np.argmax(self.plr[self.onset_idx :] < threshold)
        lidx += self.onset_idx
        return lidx

    def latency_idx_b(self) -> int:
        """Return the index of peak negative acceleration in the second after
        light onset.
        """
        acc = self.acceleration_profile()
        lidx = np.argmin(
            acc[self.onset_idx : self.onset_idx + self.sample_rate]
        )
        lidx += self.onset_idx
        return lidx

    def latency_to_constriction_a(self) -> float:
        """Return the time in miliseconds between stimulus onset and the first
        sample where constriction exceeds a percentage of the baseline, using
        the percent change threshold.
        """
        lidx = self.latency_idx_a()
        return (lidx - self.onset_idx) * (1 / self.sample_rate)

    def latency_to_constriction_b(self) -> float:
        """Return the time in miliseconds between stimulus onset and the time
        at which the pupil reaches maximal negative acceleration within a
        1-s window. See Bergamin & Kardon (2003) for justification. Requires
        well-smoothed pupil data.
        """
        lidx = self.latency_idx_b()
        return (lidx - self.onset_idx) * (1 / self.sample_rate)

    def time_to_max_constriction(self) -> float:
        """Return the time in miliseconds between stimulus onset and the peak
        of pupil constriction.
        """
        return np.argmin(self.plr[self.onset_idx :]) * (1 / self.sample_rate)

    def time_to_max_velocity(self) -> float:
        """Return the time between stimulus onset and when pupil constriction
        reaches maximum velocity.
        """
        vel = self.velocity_profile()
        return np.argmin(vel[self.onset_idx :]) * (1 / self.sample_rate)

    def peak_constriction_idx(self) -> int:
        """Return the index of the sample with peak constriction."""
        return np.argmin(self.plr)

    def peak_constriction(self) -> float:
        """Return the peak constriction value (i.e., the smallest pupil size)."""
        return np.min(self.plr)

    def constriction_amplitude(self) -> float:
        """Return the constriction amplitude (i.e. the absolute difference
        between baseline and peak constriction).
        """
        peak = self.peak_constriction()
        base = self.baseline()
        return abs(peak - base)

    def average_constriction_velocity(self) -> float:
        """Return the average constriction velocity."""
        vel = self.velocity_profile()
        lidx = self.latency_idx_a()
        pidx = self.peak_constriction_idx()
        return np.mean(abs(vel[lidx:pidx]))

    def max_constriction_velocity(self) -> float:
        """Return the maximum constriction velocity."""
        vel = self.velocity_profile()
        pidx = self.peak_constriction_idx()
        return np.max(abs(vel[self.onset_idx : pidx]))

    def max_constriction_acceleration(self) -> float:
        """Return the maximum constriction acceleration."""
        acc = self.acceleration_profile()
        pidx = self.peak_constriction_idx()
        return np.max(abs(acc[self.onset_idx : pidx]))

    def constriction_time(self) -> float:
        """Return the time difference between constriction latency and peak
        constriction.
        """
        lat = self.latency_to_constriction_a()
        ttmc = self.time_to_max_constriction()
        return ttmc - lat

    def average_redilation_velocity(self) -> float:
        """Return the average redilation velocity"""
        vel = self.velocity_profile()
        pidx = self.peak_constriction_idx()
        return np.mean(abs(vel[pidx:]))

    def max_redilation_velocity(self) -> float:
        """Return the maximum redilation velocity."""
        vel = self.velocity_profile()
        pidx = self.peak_constriction_idx()
        return np.max(abs(vel[pidx:]))

    def max_redilation_acceleration(self) -> float:
        """Return the maximum redilation acceleration."""
        acc = self.acceleration_profile()
        pidx = self.peak_constriction_idx()
        return np.max(abs(acc[pidx:]))

    def time_to_75pc_recovery(self) -> float:
        """Return the time in ms until 75% recovery from baseline."""
        base = self.baseline()
        pidx = self.peak_constriction_idx()
        amp = self.constriction_amplitude()
        return np.argmax(self.plr[pidx:] > base - (amp / 4)) * (
            1 / self.sample_rate
        )
    
    def time_to_50pc_recovery(self) -> float:
        base = self.baseline()
        pidx = self.peak_constriction_idx()
        amp = self.constriction_amplitude()
        return np.argmax(self.plr[pidx:] > base - (amp / 2)) * (1 / self.sample_rate)
    
    def time_to_25pc_recovery(self) -> float:
        base = self.baseline()
        pidx = self.peak_constriction_idx()
        amp = self.constriction_amplitude()
        return np.argmax(self.plr[pidx:] > base - (amp * 3/4)) * ( 1/ self.sample_rate)
    


    def parameters(self) -> pd.DataFrame:
        """Collapse a PLR into descriptive parameters.
        Parameters
        ----------
        plr : array-like
            Data representing a pupil's response to light in mm units.
        sample_rate : int
            sampling rate of the measurement system.
        onset_idx : int
            index of the onset of the light stimulus.
        Returns
        -------
        params : pd.DataFrame
            DataFrame containins the params.
        """

        params = {
            "Baseline": self.baseline(),
            "Latency_a": self.latency_to_constriction_a(),
            "Latency_b": self.latency_to_constriction_b(),
            "T2MaxVel": self.time_to_max_velocity(),
            "T2MaxCon": self.time_to_max_constriction(),
            "T2Rec75pc": self.time_to_75pc_recovery(),
            "PeakCon": self.peak_constriction(),
            "ConAmplitude": self.constriction_amplitude(),
            "VelConMax": self.max_constriction_velocity(),
            "VelConAve": self.average_constriction_velocity(),
            "AccConMax": self.max_constriction_acceleration(),
            "ConTime": self.constriction_time(),
            "VelRedAve": self.average_redilation_velocity(),
            "VelRedMax": self.max_redilation_velocity(),
            "AccRedMax": self.max_redilation_acceleration(),
            "50pcRecovery": self.time_to_50pc_recovery(),
            "25pcRecovery": self.time_to_25pc_recovery()
        }
        
        
        return pd.DataFrame.from_dict(params, orient="index", columns=["value"])

    def plot(
        self, vel: bool = True, acc: bool = True, print_params: bool = True
    ) -> plt.Figure:
        """Plot a PLR with option to add descriptive parameters and velocity /
        acceleration profiles.
        Parameters
        ----------
        vel_acc : bool, optional
            Whether to also plot the velocity and acceleration profiles.
            The default is False.
        print_params : bool, optional
            Whether to annotate the axis with the results of a call to
            `.parameters(...)`. The default is True.
        Returns
        -------
        fig : matplotlib.figure.Figure
            The plot.
        """
        t_max = len(self.plr) / self.sample_rate
        time = np.linspace(0, t_max, num=len(self.plr)) - (
            self.onset_idx / self.sample_rate
        )
        b = self.baseline()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(time, self.plr, lw=4)
        ax.axhline(b, 0, 1, ls="dashed", color="k", lw=1)
        ax.axvspan(0, 0 + self.stim_duration, color="k", alpha=0.3)
        ax.set_ylabel("Pupil Size")
        ax.set_xlabel("Time (s)")

        if vel or acc:
            ax2 = ax.twinx()
            if vel:
                vel = self.velocity_profile()
                ax2.plot(time, vel, color="g", lw=2.5)
            if acc:
                acc = self.acceleration_profile()
                ax2.plot(time, acc, color="r", lw=1)
            ax2.set_ylabel("Velocity / Acceleration")

        if print_params:
            params = self.parameters()
            params = params.round(3)
            ax.text(
                0.78,
                0.03,
                params.to_string(header=False, justify="right"),
                size=8,
                transform=ax.transAxes,
            )

        return fig
        
    def mask_pupil_first_derivative(
        samples: pd.DataFrame,
        threshold: float = 3.0,
        mask_cols: List[str] = ["diameter"],
    ) -> pd.DataFrame:
        """Apply a masking threshold on the first derivative of pupil data.
        Use a statistical criterion on the first derivative of pupil data to mask
        poor quality data. Helpful for dealing with blinks.
        Parameters
        ----------
        samples : pandas.DataFrame
            Samples containing the data to be masked.
        threshold : float, optional
            Number of standard deviations from the mean of the first derivative
            to use as the threshold for masking. The default is 3.0.
        mask_cols : list, optional
            Columns to mask. The default is ``['diameter']``.
        Returns
        -------
        samps : pandas.DataFrame
            masked data
        """
        samps = samples.copy(deep=True)
        for col in mask_cols:
            d = samps[col].diff()
            m = samps[col].diff().mean()
            s = samps[col].diff().std() * threshold
            # TODO: check this works properly
            samps[col] = samps[col].mask((d < (m - s)) | (d > (m + s)))
            samps.loc[samps[col] == 0, col] = np.nan
        return samps
    
    
    def mask_pupil_confidence(
        samples: pd.DataFrame,
        threshold: float = 0.8,
        mask_cols: List[str] = ["diameter"],
    ) -> pd.DataFrame:
        """
        Sets data in mask_cols to NaN where the corresponding confidence metric is
        below threshold. Pupil Labs reccommend a threshold of 0.8. Helpful for
        dealing with blinks.
        Parameters
        ----------
        samples : pandas.DataFrame
            Samples containing the data to be masked.
        threshold : float, optional
            Confidence threshold for masking. The default is 0.8.
        mask_cols : list, optional
            Columns to mask. The default is ['diameter'].
        Returns
        -------
        samps : pandas.DataFrame
            masked data
        """
        samps = samples.copy(deep=True)
        samps[mask_cols] = samps[mask_cols].mask(samps.confidence < threshold)
        return samps
    
    def interpolate_pupil(
        samples: pd.DataFrame,
        interp_cols: List[str] = ["diameter"],
        method: str = "linear",
        order: int = None,
    ) -> pd.DataFrame:
        """
        Use linear interpolation to reconstruct nan values in interp_cols.
        Parameters
        ----------
        samples : pandas.DataFrame
            Samples containing the data to be interpolated.
        interp_cols : list, optional
            Columns to interpolate. The default is ['diameter'].
        method : string
            Linear or polynomial. If polynomial, requires 'order' to be specified.
        order : int
            Polynomial order.
        Returns
        -------
        samps : pandas.DataFrame
            Masked data
        """
        if method == "polynomial" and not order:
            raise ValueError("Must specify order for polynomial")
        samps = samples.copy(deep=True)
        samps["interpolated"] = 0
        samps.loc[samps[interp_cols].isna().any(axis=1), "interpolated"] = 1
        samps[interp_cols] = samps[interp_cols].interpolate(
            method=method,
            order=order,
            axis=0,
            inplace=False,
            limit=10000,
            limit_direction="both",
        )
        return samps
    
    def percent_signal_change(
        ranges: pd.DataFrame, baselines: pd.DataFrame, pupil_cols: List[str] = []
    ) -> pd.DataFrame:
        """Add new columns expressing pupil_cols as percent change from baseline.
        Parameters
        ----------
        ranges : pandas.DataFrame
            DataFrame with MultiIndex, e.g., output from ``utils.extract(...)``
        baselines : pandas.DataFrame
            DataFrame with baselines.
        pupil_cols : list, optional
            List of pupil columns to process. The default is [].
        Returns
        -------
        ranges : pandas.DataFrame
            Original DataFrame with new columns for percent-signal change.
        """
        for col in pupil_cols:
            new_col = col + "_pc"
            ranges[new_col] = (ranges[col] / baselines[col] - 1).values * 100
        return ranges
    
    def mask_zeros(
        samples: pd.DataFrame, mask_cols: List[str] = ["diameter"]
    ) -> pd.DataFrame:
        """Sets any 0 values in columns in mask_cols to NaN.
        Parameters
        ----------
        samples : pandas.DataFrame
            The samples to search for 0 values.
        mask_cols (list of strings)
            The columns to search for 0 values.
        Returns
        -------
        samps : pandas.DataFrame
            Masked data
        """
        samps = samples.copy(deep=True)
        for f in mask_cols:
            samps[samps[f] == 0] = float("nan")
        return samps
    
    
    def interpolate_zeros(
        samples: pd.DataFrame, fields: List[str] = ["diameter"]
    ) -> pd.DataFrame:
        """Replace 0s in "samples" with linearly interpolated data.
        Parameters
        ----------
        samples : pandas.DataFrame
            The samples in which you'd like to replace 0s
        interp_cols : list
            The column names from samples in which you'd like to replace 0s.
        """
        samps = PLR.mask_zeros(samples, mask_cols=fields)
        samps = samps.interpolate(method="linear", axis=0, inplace=False)
        # since interpolate doesn't handle the start/finish, bfill the ffill to
        # take care of NaN's at the start/finish samps.
        samps.fillna(method="bfill", inplace=True)
        samps.fillna(method="ffill", inplace=True)
        return samps


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:45:28 2023

@author: Katharina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyplr import utils
from pyplr import graphing as graphing
from pyplr import preproc
import matplotlib.pyplot as plt 
import neurokit2 as nk
import os
import scipy.stats as stats
from typing import List
from scipy import signal



class PLR2d:
    """Class to handle data representing a pupil response to a flash of light."""

    # TODO: add time stuff
    def __init__(
        self,
        plr: object,
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
    
    def mask_pupil_zscore(
            df_list: List[pd.DataFrame],
            threshold: float = 3.0,
            mask_cols: List[str] = ["diameter"],
        ) -> List[pd.DataFrame]:
            """
            Apply a masking threshold on the z-score of pupil data.
            Use a statistical criterion on the z-score of pupil data to mask poor quality data.
            Helpful for dealing with extreme values due to pupil dilation or constriction.
        
            Parameters
            ----------
            df_list : List[pandas.DataFrame]
                List of dataframes containing the data to be masked.
            threshold : float, optional
                Number of standard deviations from the mean to use as the threshold for masking.
                The default is 3.0.
            mask_cols : list, optional
                Columns to mask. The default is ``['diameter_3d']``.
        
            Returns
            -------
            preprocessed_dfs : List[pandas.DataFrame]
                Preprocessed data in the form of a list of dataframes
            """
            preprocessed_dfs = []
            for df in df_list:
                df_copy = df.copy(deep=True)
                for col in mask_cols:
                    diameter_zscore = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
                    diameter_mask = abs(diameter_zscore) > threshold
                    df_copy.loc[diameter_mask, col] = np.nan
                preprocessed_dfs.append(df_copy)
            return preprocessed_dfs
     
        
    def mask_pupil_first_derivative(
        df_list: List[pd.DataFrame],
        threshold: float = 3.0,
        mask_cols: List[str] = ["diameter"],
    ) -> List[pd.DataFrame]:
        """
        Apply a masking threshold on the first derivative of pupil data.
        Use a statistical criterion on the first derivative of pupil data to mask poor quality data.
        Helpful for dealing with blinks.

        Parameters
        ----------
        df_list : List[pandas.DataFrame]
            List of dataframes containing the data to be masked.
        threshold : float, optional
            Number of standard deviations from the mean of the first derivative to use as the threshold for masking.
            The default is 3.0.
        mask_cols : list, optional
            Columns to mask. The default is ``['diameter']``.

        Returns
        -------
        preprocessed_dfs : List[pandas.DataFrame]
            Preprocessed data in the form of a list of dataframes
        """
        preprocessed_dfs = []
        for df in df_list:
            df_copy = df.copy(deep=True)
            for col in mask_cols:
                d = df_copy[col].diff()
                m = df_copy[col].diff().mean()
                s = df_copy[col].diff().std() * threshold
                mask = (d < (m - s)) | (d > (m + s))
                df_copy.loc[mask, col] = np.nan
            preprocessed_dfs.append(df_copy)
        return preprocessed_dfs


    def mask_pupil_confidence(
        df_list: List[pd.DataFrame],
        threshold: float,
        mask_cols: List[str] = ["diameter"],
    ) -> List[pd.DataFrame]:
        """
        Sets data in mask_cols to NaN where the corresponding confidence metric is
        below threshold. Pupil Labs reccommend a threshold of 0.8. Helpful for
        dealing with blinks.
        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of dataframes containing the data to be masked.
        threshold : float, optional
            Confidence threshold for masking. The default is 0.8.
        mask_cols : list, optional
            Columns to mask. The default is ['diameter'].
        Returns
        -------
        preprocessed_dfs : List[pd.DataFrame]
            Preprocessed data in the form of a list of dataframes
        """
        preprocessed_dfs = []
        for df in df_list:
            df_copy = df.copy(deep=True)
            mask = (df_copy["confidence"] < threshold)
            df_copy.loc[mask, mask_cols] = np.nan
            preprocessed_dfs.append(df_copy)
        return preprocessed_dfs

   
        
    def remove_threshold(
        df_list: List[pd.DataFrame],
        lower_threshold: float,
        upper_threshold: float,
        mask_cols: List[str]
    ) -> List[pd.DataFrame]:
        """
        Set data in mask_cols to NaN where the corresponding values are outside of the lower and upper thresholds.
        
        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of dataframes containing the data to be masked.
        lower_threshold : float
            Lower threshold value for masking.
        upper_threshold : float
            Upper threshold value for masking.
        mask_cols : List[str]
            Columns to mask.
        
        Returns
        -------
        preprocessed_dfs : List[pd.DataFrame]
            Preprocessed data in the form of a list of dataframes
        """
        
        preprocessed_dfs = []
        for df in df_list:
            df_copy = df.copy(deep=True)
            for col in mask_cols:
                df_copy[col] = df_copy[col].where(
                    (df_copy[col] >= lower_threshold) & (df_copy[col] <= upper_threshold),
                    np.nan
                )
            preprocessed_dfs.append(df_copy)
    
        return preprocessed_dfs
    
 
    

    def iqr_threshold(
        df_list: List[pd.DataFrame],
        iqr_factor: float,
        mask_cols: List[str]
    ) -> List[pd.DataFrame]:
        """
        Mask data points with values outside of the IQR range in each column specified in mask_cols.
        
        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of dataframes containing the data to be processed.
        iqr_factor : float
            Factor by which to multiply the IQR range to determine the outlier threshold.
        mask_cols : List[str]
            Columns to process.
        
        Returns
        -------
        preprocessed_dfs : List[pd.DataFrame]
            Preprocessed data in the form of a list of dataframes.
        """
        
        preprocessed_dfs = []
        for df in df_list:
            df_copy = df.copy(deep=True)
            for col in mask_cols:
                q1 = df_copy[col].quantile(0.25)
                q3 = df_copy[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                df_copy[col] = df_copy[col].mask(
                    (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound),
                    other=np.nan
                )
            preprocessed_dfs.append(df_copy)
        
        return preprocessed_dfs
    
    
    def detect_large_oscillations(self, threshold: float) -> None:
        """
        Detects large oscillations in pupil size data using the power spectrum.

        Parameters:
            threshold (float): Threshold value for detecting large oscillations.

        Returns:
            None
        """
        fs = self.sample_rate

        # Compute the power spectrum using FFT
        spectrum = np.abs(fft(self.plr['pupil_size']))

        # Calculate the frequency axis
        freq_axis = np.fft.fftfreq(len(self.plr['pupil_size']), 1 / fs)

        # Find the maximum amplitude in the power spectrum
        max_amplitude = np.max(spectrum)

        # Find the frequency associated with the maximum amplitude
        max_amplitude_freq = freq_axis[np.argmax(spectrum)]

        # Check if the maximum amplitude exceeds the threshold
        large_oscillations_mask = max_amplitude > threshold

        # Set the 'pupil_size' values to NaN where large oscillations are detected
        self.plr.loc[large_oscillations_mask, 'pupil_size'] = np.nan
        
    def detect_large_oscillations(df_list: List[pd.DataFrame], fs: float, threshold: float) -> List[pd.DataFrame]:
        """
        Detects large oscillations in pupil size data using the power spectrum.

        Args:
            df_list (List[pd.DataFrame]): List of dataframes containing the pupil size data.
            fs (float): Sampling frequency of the data.
            threshold (float): Threshold value for detecting large oscillations.

        Returns:
            List[pd.DataFrame]: List of dataframes with large oscillations removed.
        """

        preprocessed_dfs = []
        for df in df_list:
            df_copy = df.copy(deep=True)

            # Apply detect_large_oscillations to the 'pupil_size' column of each dataframe
            large_oscillations_mask = detect_large_oscillations(df_copy['diameter'], fs, threshold)

            # Set the 'pupil_size' values to NaN where large oscillations are detected
            df_copy.loc[large_oscillations_mask, 'diameter'] = np.nan

            preprocessed_dfs.append(df_copy)

        return preprocessed_dfs

    
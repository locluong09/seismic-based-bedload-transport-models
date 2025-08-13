from abc import ABC, abstractmethod

class SeismicBasedBedloadTransportModel(ABC):
    """
    Abstract base class for seismic-based bedload transport models.
    """

    @abstractmethod
    def forward_psd(self, frequency, grain_size, flow_depth, channel_width, slope_angle, source_receiver_distance, qb, **kwargs):
        """
        Calculate the power spectral density (PSD) for sediment transport.

        Args:
            frequency : Frequency window (Hz).
            grain_size : Grain size diameter (m).
            flow_depth : Water depth (m).
            channel_width : Channel width (m).
            slope_angle : Channel slope (-).
            source_receiver_distance : Distance from the river thalweg to a seismic sensor (m).
        
        Returns:
            Power-spectral density (PSD) of seismic velocity signal.
        """
        pass

    @abstractmethod
    def inverse_bedload(self, frequency, grain_size, flow_depth, channel_width, slope_angle, source_receiver_distance, qb, **kwargs):
        """
        Inverse calculation for bedload transport based on seismic data.

        Args:
            frequency : Frequency window (Hz).
            grain_size : Grain size diameter (m).
            flow_depth : Water depth (m).
            channel_width : Channel width (m).
            slope_angle : Channel slope (-).
            source_receiver_distance : Distance from the river thalweg to a seismic sensor (m).

        Returns:
            Inverse back bedload flux from seismic PSD.
        """
        pass

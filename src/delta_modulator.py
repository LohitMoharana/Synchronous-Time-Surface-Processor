# src/delta_modulator.py

def analog_to_spikes(analog_data, sampling_rate_hz, threshold_mv):
    """
    Converts continuous analog multi-channel data into discrete AER spikes.

    analog_data: A 2D list/array. Rows = time samples, Columns = sensor channels (e.g., 64 channels for an 8x8 EMG patch).
    sampling_rate_hz: The frequency the analog data was recorded at (e.g., 2000 Hz).
    threshold_mv: The Delta-V threshold required to trigger a spike.
    """
    events = []

    # Calculate how many microseconds pass between each analog sample
    sample_period_us = (1.0 / sampling_rate_hz) * 1_000_000

    num_samples = len(analog_data)
    num_channels = len(analog_data[0]) if num_samples > 0 else 0

    # Track the last known voltage for every sensor channel
    last_voltages = [analog_data[0][ch] for ch in range(num_channels)]

    print(f"[Delta Modulator] Converting {num_samples} analog samples across {num_channels} channels to spikes...")

    for t_index in range(1, num_samples):
        current_time_us = t_index * sample_period_us

        for ch in range(num_channels):
            current_v = analog_data[t_index][ch]
            delta_v = current_v - last_voltages[ch]

            # Did the voltage cross the threshold?
            if abs(delta_v) >= threshold_mv:
                # Map the 1D channel index back to a 2D 8x8 Hardware Grid
                # e.g., Channel 15 -> X=7, Y=1
                x = ch % 8
                y = ch // 8

                # Determine Polarity (1 = Voltage went UP, 0 = Voltage went DOWN)
                polarity = 1 if delta_v > 0 else 0

                events.append({
                    'x': x,
                    'y': y,
                    't': current_time_us,
                    'p': polarity
                })

                # Update the baseline to the new voltage
                last_voltages[ch] = current_v

    print(f"[Delta Modulator] Conversion complete! Generated {len(events)} discrete spikes.")
    return events
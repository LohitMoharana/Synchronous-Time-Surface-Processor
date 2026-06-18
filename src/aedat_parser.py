# src/aedat_parser.py
import struct
import os


def load_aedat_file(filepath):
    """
    Parses a raw AEDAT 2.0 binary file from a DVS128 camera.
    Returns a list of dictionaries: [{'x', 'y', 't', 'p'}, ...]
    """
    events = []

    print(f"[Binary Parser] Opening {os.path.basename(filepath)}...")

    with open(filepath, 'rb') as f:
        # 1. Skip the Header (lines starting with '#')
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)  # Go back, the header is over
                break

        # 2. Read the raw binary data in 8-byte chunks (AEDAT 2.0 format)
        while True:
            data = f.read(8)
            if not data or len(data) != 8:
                break

            # Unpack 2 unsigned 32-bit integers (Big Endian)
            address, timestamp = struct.unpack('>II', data)

            # 3. Bitmasking to extract physical coordinates (DVS128 Protocol)
            x_addr = (address >> 1) & 0x7F
            y_addr = (address >> 8) & 0x7F
            polarity = (address >> 0) & 1

            events.append({
                'x': x_addr,
                'y': y_addr,
                't': float(timestamp),
                'p': polarity
            })

    print(f"[Binary Parser] Extracted {len(events)} raw events.")
    return events
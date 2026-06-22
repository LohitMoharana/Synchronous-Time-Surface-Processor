import math

tau_us = 75_000.0

print("Checking different decay periods:\n")
for T_us in [150, 200, 300, 500, 1000, 2000, 5000]:
    T_cycles = int(T_us / 0.02)  # cycles at 50MHz
    factor_float = math.exp(-T_us / tau_us)
    factor_8bit = round(factor_float * 256)

    if factor_8bit >= 256:
        print(f"T={T_us}us ({T_cycles} cycles): factor rounds to {factor_8bit}/256 -- TOO LARGE, skip")
        continue

    actual_factor = factor_8bit / 256
    effective_tau = -T_us / math.log(actual_factor)

    # Simulate decay from 255 with 16-bit internal storage
    val = 255 << 8
    errors = []
    for tick in range(1, 20):
        val = (val * factor_8bit) >> 8
        t = tick * T_us
        py_val = int(255 * math.exp(-t / tau_us))
        rtl_val = val >> 8
        errors.append(abs(rtl_val - py_val))

    print(f"T={T_us}us ({T_cycles} cycles): "
          f"factor={factor_8bit}/256, "
          f"eff_tau={effective_tau:.0f}us, "
          f"max_err={max(errors)} LSB")
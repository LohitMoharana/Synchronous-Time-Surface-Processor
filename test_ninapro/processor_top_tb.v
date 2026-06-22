module processor_top_tb;
    reg clk;
    reg rst_n;
    reg event_valid;
    reg [2:0] event_x;
    reg [2:0] event_y;
    reg event_pol;

    wire sys_en;
    wire decay_tick;
    wire [511:0] flat_surface;

    processor_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .event_valid(event_valid),
        .event_x(event_x),
        .event_y(event_y),
        .event_pol(event_pol),
        .sys_en(sys_en),
        .decay_tick(decay_tick),
        .flat_surface(flat_surface)
    );

    // 50 MHz clock (20ns period)
    initial clk = 0;
    always #10 clk = ~clk;

    integer stimulus_file, output_file;
    integer wait_cycles, pol, x, y;
    integer scan_status;
    integer i, r, c;
    integer event_index, current_cycle;
    real time_us;

    reg [7:0] matrix_2d [0:7][0:7];

    initial begin
        stimulus_file = $fopen("synthetic_data.txt", "r");
        output_file   = $fopen("matrix_history.txt", "w");

        if (stimulus_file == 0) begin
            $display("ERROR: Could not open synthetic_data.txt.");
            $finish;
        end

        rst_n         = 0;
        event_valid   = 0;
        event_index   = 0;
        current_cycle = 0;

        #105;
        rst_n = 1;

        $display("Starting RTL Simulation. Please wait...");

        while (!$feof(stimulus_file)) begin
            scan_status = $fscanf(stimulus_file, "%d %d %d %d\n", wait_cycles, pol, x, y);

            if (scan_status == 4) begin

                if (event_index % 1000 == 0 && event_index > 0)
                    $display("Processed %0d spikes... (cycle %0d)", event_index, current_cycle);

                // Enforce minimum 1-cycle gap even when wait_cycles == 0.
                // Without this, back-to-back injections give the decay engine
                // zero clock cycles to tick between events.
                if (wait_cycles < 1)
                    wait_cycles = 1;

                for (i = 0; i < wait_cycles; i = i + 1) begin
                    @(posedge clk);
                    current_cycle = current_cycle + 1;
                end

                // Inject spike
                event_valid = 1;
                event_pol   = pol;
                event_x     = x[2:0];
                event_y     = y[2:0];

                @(posedge clk);
                current_cycle = current_cycle + 1;
                event_valid = 0;

                // Capture and log matrix
                for (r = 0; r < 8; r = r + 1)
                    for (c = 0; c < 8; c = c + 1)
                        matrix_2d[r][c] = flat_surface[((r*8 + c)*8) +: 8];

                time_us = $itor(current_cycle) * 0.02;
                $fwrite(output_file, "EVENT_INDEX: %0d | TIME_US: %0.1f\n", event_index, time_us);

                for (r = 7; r >= 0; r = r - 1) begin
                    $fwrite(output_file, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                        matrix_2d[r][0], matrix_2d[r][1], matrix_2d[r][2], matrix_2d[r][3],
                        matrix_2d[r][4], matrix_2d[r][5], matrix_2d[r][6], matrix_2d[r][7]);
                end
                $fwrite(output_file, "---\n");

                event_index = event_index + 1;
            end
        end

        $fclose(stimulus_file);
        $fclose(output_file);
        $display("[SUCCESS] Simulation complete. Output: matrix_history.txt");
        $finish;
    end
endmodule
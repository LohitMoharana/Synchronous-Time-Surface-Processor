`timescale 1ns / 1ps

module processor_top_tb;
    reg clk;
    reg rst_n;
    reg event_valid;
    reg [2:0] event_x;
    reg [2:0] event_y;
    reg event_pol;
    reg window_start;
    reg [2:0] snap_addr;

    wire        sys_en;
    wire        decay_tick;
    wire [511:0] flat_surface;
    wire [511:0] snap_data;
    wire         features_ready;

    processor_top_corrected uut (
        .clk(clk),
        .rst_n(rst_n),
        .event_valid(event_valid),
        .event_x(event_x),
        .event_y(event_y),
        .event_pol(event_pol),
        .window_start(window_start),
        .snap_addr(snap_addr),
        .sys_en(sys_en),
        .decay_tick(decay_tick),
        .flat_surface(flat_surface),
        .snap_data(snap_data),
        .features_ready(features_ready)
    );

    // 50MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    integer stimulus_file, output_file;
    integer wait_cycles, pol, x, y, is_boundary;
    integer scan_status;
    integer event_index, current_cycle;
    integer r, c, s, i;
    real    time_us;

    reg [7:0] matrix_2d [0:7][0:7];
    reg features_ready_prev;
    integer snapshot_count;

    // --------------------------------------------------------
    // VCD dump for power analysis
    // Only dump first 10000 events to keep file manageable
    // This covers several gesture trials with real switching
    // --------------------------------------------------------
    integer vcd_event_limit;
    initial begin
        vcd_event_limit = 10000;
        $dumpfile("D:/RTL_OUT/power_sim.vcd");
        $dumpvars(0, processor_top_tb.uut);;
    end

    initial begin
        stimulus_file = $fopen("D:/RTL_OUT/synthetic_data_short.txt", "r");
        output_file   = $fopen("D:/RTL_OUT/matrix_history.txt", "w");

        if (stimulus_file == 0) begin
            $display("ERROR: Cannot open synthetic_data.txt");
            $finish;
        end

        rst_n               = 0;
        event_valid         = 0;
        window_start        = 0;
        snap_addr           = 3'd0;
        event_index         = 0;
        current_cycle       = 0;
        features_ready_prev = 0;
        snapshot_count      = 0;

        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        window_start = 1;
        @(posedge clk);
        current_cycle = current_cycle + 1;
        window_start = 0;

        $display("Simulation started...");

        while (!$feof(stimulus_file)) begin
            scan_status = $fscanf(stimulus_file,
                "%d %d %d %d %d\n",
                wait_cycles, pol, x, y, is_boundary);

            if (scan_status == 5) begin

                if (event_index % 1000 == 0 && event_index > 0)
                    $display("Events: %0d, Cycle: %0d, sys_en=%b",
                             event_index, current_cycle, sys_en);

                // Fixed cycle loop - no variable delay
                // Vivado compatible
                for (i = 0; i < wait_cycles; i = i + 1) begin
                    @(posedge clk);
                    current_cycle = current_cycle + 1;
                end

                // Assert window_start on trial boundary
                if (is_boundary == 1) begin
                    window_start = 1;
                    @(posedge clk);
                    current_cycle = current_cycle + 1;
                    window_start = 0;
                end

                // Check features_ready transition
                if (features_ready && !features_ready_prev) begin
                    snapshot_count = snapshot_count + 1;
                    $display("Window %0d complete at cycle %0d",
                             snapshot_count, current_cycle);
                    $fwrite(output_file,
                        "SNAPSHOT_WINDOW: %0d | CYCLE: %0d\n",
                        snapshot_count, current_cycle);
                    for (s = 0; s < 8; s = s + 1) begin
                        snap_addr = s;
                        @(posedge clk);
                        current_cycle = current_cycle + 1;
                        $fwrite(output_file,
                            "SNAP_%0d: %h\n", s, snap_data);
                    end
                    $fwrite(output_file, "---\n");
                end
                features_ready_prev = features_ready;

                // Inject event
                event_valid = 1;
                event_pol   = pol;
                event_x     = x[2:0];
                event_y     = y[2:0];

                @(posedge clk);
                current_cycle = current_cycle + 1;
                event_valid = 0;

                // Log matrix
                for (r = 0; r < 8; r = r + 1)
                    for (c = 0; c < 8; c = c + 1)
                        matrix_2d[r][c] =
                            flat_surface[((r*8+c)*8) +: 8];

                time_us = $itor(current_cycle) * 0.02;
                $fwrite(output_file,
                    "EVENT_INDEX: %0d | TIME_US: %0.1f\n",
                    event_index, time_us);

                for (r = 7; r >= 0; r = r - 1)
                    $fwrite(output_file,
                        "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                        matrix_2d[r][0], matrix_2d[r][1],
                        matrix_2d[r][2], matrix_2d[r][3],
                        matrix_2d[r][4], matrix_2d[r][5],
                        matrix_2d[r][6], matrix_2d[r][7]);
                $fwrite(output_file, "---\n");

                event_index = event_index + 1;

                // Stop VCD dump after limit
                // Full simulation continues but VCD stops
                // This keeps file size manageable for power import
                if (event_index == vcd_event_limit) begin
                    $dumpoff;
                    $display("VCD dump stopped at %0d events",
                             vcd_event_limit);
                end
            end
        end

        $fclose(stimulus_file);
        $fclose(output_file);
        $display("Done. Events: %0d, Windows: %0d",
                 event_index, snapshot_count);
        $finish;
    end
endmodule
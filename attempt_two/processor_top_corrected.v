`timescale 1ns / 1ps

// ============================================================
// MODULE 1: Activity Monitor
// Tracks idle time. sys_en goes low after TIMEOUT_CYCLES
// of no events, gating downstream power consumption.
// ============================================================
module activity_monitor #(
    parameter TIMEOUT_CYCLES = 5000
)(
    input  wire clk,
    input  wire rst_n,
    input  wire event_valid,
    output wire sys_en
);
    reg [31:0] idle_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            idle_counter <= 32'd0;
        else if (event_valid)
            idle_counter <= 32'd0;
        else if (idle_counter < TIMEOUT_CYCLES)
            idle_counter <= idle_counter + 1;
    end

    assign sys_en = (idle_counter < TIMEOUT_CYCLES);
endmodule


// ============================================================
// MODULE 2: Decay Engine
// Free-running timer. Fires decay_tick every DECAY_PERIOD
// cycles regardless of sys_en.
// DECAY_PERIOD = 15000 cycles = 300us at 50MHz
// Matched to Python: DECAY_FACTOR=255, eff_tau=76.6ms
// ============================================================
module decay_engine #(
    parameter DECAY_PERIOD = 15000
)(
    input  wire clk,
    input  wire rst_n,
    input  wire sys_en,
    output wire decay_tick
);
    reg [31:0] timer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            timer <= 32'd0;
        else begin
            if (timer == DECAY_PERIOD - 1)
                timer <= 32'd0;
            else
                timer <= timer + 1;
        end
    end

    assign decay_tick = (timer == DECAY_PERIOD - 1);
endmodule


// ============================================================
// MODULE 3: Time Surface Array
// 8x8 grid of 16-bit fixed-point decay registers.
// Injection sets cell to 255<<8 = 65280.
// Decay: cell = (cell * DECAY_FACTOR) >> 8
// Output: upper 8 bits of each cell = actual value 0-255.
// DECAY_FACTOR=255 matched to Python RTL model exactly.
// ============================================================
module time_surface_array #(
    parameter DECAY_FACTOR = 255
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        event_valid,
    input  wire [2:0]  event_x,
    input  wire [2:0]  event_y,
    input  wire        decay_tick,
    output wire [511:0] flat_surface
);
    reg [15:0] surface_fp [0:7][0:7];
    integer r, c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (r = 0; r < 8; r = r + 1)
                for (c = 0; c < 8; c = c + 1)
                    surface_fp[r][c] <= 16'd0;
        end else begin
            for (r = 0; r < 8; r = r + 1) begin
                for (c = 0; c < 8; c = c + 1) begin
                    if (event_valid &&
                        (event_x == c) && (event_y == r))
                        // Injection: 255 in Q8.8 fixed point
                        surface_fp[r][c] <= 16'd65280;
                    else if (decay_tick)
                        // Multiply-shift decay matching Python
                        surface_fp[r][c] <=
                            (surface_fp[r][c] * DECAY_FACTOR) >> 8;
                end
            end
        end
    end

    // Output upper 8 bits of each cell
    genvar gx, gy;
    generate
        for (gy = 0; gy < 8; gy = gy + 1) begin : gen_y
            for (gx = 0; gx < 8; gx = gx + 1) begin : gen_x
                assign flat_surface[((gy*8+gx)*8) +: 8] =
                    surface_fp[gy][gx][15:8];
            end
        end
    endgenerate
endmodule


// ============================================================
// MODULE 4: Snapshot Controller
// Takes 8 evenly spaced snapshots of flat_surface during
// a gesture trial window.
// SNAPSHOT_INTERVAL = 35,000,000 cycles = 700ms at 50MHz
// Matches Python: 8 snapshots across ~5600ms trial duration
// feature_vector = 8 x 512 bits = 4096 bits total
// ============================================================
// ============================================================
// MODULE 4: Snapshot Controller with BRAM storage
// Stores 8 snapshots of flat_surface during a gesture window.
// Read interface: present snap_addr (0-7), read snap_data
// after one clock cycle (synchronous BRAM read).
// This pattern reliably infers BRAM on Artix-7 in Vivado.
// ============================================================
module snapshot_controller #(
    parameter SNAPSHOT_INTERVAL = 35_000_000,
    parameter N_SNAPSHOTS       = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        window_start,
    input  wire [511:0] flat_surface,
    // Read interface
    input  wire [2:0]  snap_addr,
    output reg  [511:0] snap_data,
    output reg         features_ready
);
    // BRAM storage: 8 entries of 512 bits each
    // Vivado infers BRAM36 for this pattern
    reg [511:0] snapshot_mem [0:7];

    reg [25:0] cycle_counter;
    reg [2:0]  snapshot_count;
    reg        capturing;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter  <= 26'd0;
            snapshot_count <= 3'd0;
            features_ready <= 1'b0;
            capturing      <= 1'b0;
        end else begin
            if (window_start) begin
                cycle_counter  <= 26'd0;
                snapshot_count <= 3'd0;
                features_ready <= 1'b0;
                capturing      <= 1'b1;
            end else if (capturing) begin
                if (cycle_counter == SNAPSHOT_INTERVAL - 1) begin
                    // Write snapshot to BRAM
                    snapshot_mem[snapshot_count] <= flat_surface;
                    cycle_counter <= 26'd0;

                    if (snapshot_count == N_SNAPSHOTS - 1) begin
                        features_ready <= 1'b1;
                        capturing      <= 1'b0;
                    end else begin
                        snapshot_count <= snapshot_count + 3'd1;
                    end
                end else begin
                    cycle_counter <= cycle_counter + 26'd1;
                end
            end
        end
    end

    // Synchronous read - one cycle latency, infers BRAM
    always @(posedge clk) begin
        snap_data <= snapshot_mem[snap_addr];
    end
endmodule


// ============================================================
// MODULE 5: Processor Top Level
// ============================================================
module processor_top_corrected (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        event_valid,
    input  wire [2:0]  event_x,
    input  wire [2:0]  event_y,
    input  wire        event_pol,
    input  wire        window_start,
    input  wire [2:0]  snap_addr,
    output wire        sys_en,
    output wire        decay_tick,
    output wire [511:0] flat_surface,
    output wire [511:0] snap_data,
    output wire         features_ready
);
    activity_monitor #(
        .TIMEOUT_CYCLES(5000)
    ) monitor_inst (
        .clk(clk),
        .rst_n(rst_n),
        .event_valid(event_valid),
        .sys_en(sys_en)
    );

    decay_engine #(
        .DECAY_PERIOD(15000)
    ) decay_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_en),
        .decay_tick(decay_tick)
    );

    time_surface_array #(
        .DECAY_FACTOR(255)
    ) array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .event_valid(event_valid),
        .event_x(event_x),
        .event_y(event_y),
        .decay_tick(decay_tick),
        .flat_surface(flat_surface)
    );

    snapshot_controller #(
        .SNAPSHOT_INTERVAL(35_000_000),
        .N_SNAPSHOTS(8)
    ) snap_inst (
        .clk(clk),
        .rst_n(rst_n),
        .window_start(window_start),
        .flat_surface(flat_surface),
        .snap_addr(snap_addr),
        .snap_data(snap_data),
        .features_ready(features_ready)
    );
endmodule
module processor_top (
    input wire clk,
    input wire rst_n,
    input wire event_valid,
    input wire [2:0] event_x,
    input wire [2:0] event_y,
    input wire event_pol,
    output wire sys_en,
    output wire decay_tick,
    output wire [511:0] flat_surface
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
        .DECAY_PERIOD(50)
    ) decay_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_en),
        .decay_tick(decay_tick)
    );

    time_surface_array #(
        .DECAY_FACTOR(253)
    ) array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .event_valid(event_valid),
        .event_x(event_x),
        .event_y(event_y),
        .decay_tick(decay_tick),
        .flat_surface(flat_surface)
    );

endmodule
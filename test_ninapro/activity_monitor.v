module activity_monitor #(
    parameter TIMEOUT_CYCLES = 5000
)(
    input wire clk,
    input wire rst_n,
    input wire event_valid,
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
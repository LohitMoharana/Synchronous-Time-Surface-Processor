module decay_engine #(
    parameter DECAY_PERIOD = 50      // 50 cycles = 1us at 50MHz
)(
    input wire clk,
    input wire rst_n,
    input wire sys_en,               // unused internally; kept for top-level port
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
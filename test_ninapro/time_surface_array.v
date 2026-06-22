module time_surface_array #(
    parameter DECAY_FACTOR = 253     // 8-bit fixed-point: 253/256 ~ exp(-1us/75us)
)(
    input wire clk,
    input wire rst_n,
    input wire event_valid,
    input wire [2:0] event_x,
    input wire [2:0] event_y,
    input wire decay_tick,
    output wire [511:0] flat_surface
);
    // 16-bit internal fixed-point storage: actual_value = surface_fp >> 8
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
                    if (event_valid && (event_x == c) && (event_y == r)) begin
                        // Injection: set to 255 in fixed-point (255 << 8 = 65280)
                        surface_fp[r][c] <= 16'd65280;
                    end else if (decay_tick) begin
                        // Multiply by DECAY_FACTOR, discard lower 8 bits
                        // 16-bit * 8-bit = 24-bit product; upper 16 bits kept
                        surface_fp[r][c] <= (surface_fp[r][c] * DECAY_FACTOR) >> 8;
                    end
                end
            end
        end
    end

    // Output: upper 8 bits of each 16-bit fixed-point cell
    genvar gx, gy;
    generate
        for (gy = 0; gy < 8; gy = gy + 1) begin : gen_y
            for (gx = 0; gx < 8; gx = gx + 1) begin : gen_x
                assign flat_surface[((gy*8 + gx)*8) +: 8] = surface_fp[gy][gx][15:8];
            end
        end
    endgenerate

endmodule
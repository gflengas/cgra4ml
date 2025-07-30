# Top level rtl module
add_files  [glob $CONFIG_DIR/*.svh] [glob $RTL_DIR/*] [glob $RTL_DIR/ext/*]
update_compile_order -fileset sources_1

set_property top axi_cgra4ml [current_fileset]
create_bd_cell -type module -reference axi_cgra4ml axi_cgra4ml_0

if {$BOARD eq "pynq_z2"} {
    set_property -dict [list CONFIG.AXIL_ADDR_WIDTH {32}] [get_bd_cells axi_cgra4ml_0]

    # Create and configure the AXI SmartConnect for the HP0 port
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_hp0
    set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_hp0]

    # Connect the AXI masters from the CGRA core to the slave interfaces of the SmartConnect
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_output] [get_bd_intf_pins smartconnect_hp0/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_pixel]  [get_bd_intf_pins smartconnect_hp0/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_weights] [get_bd_intf_pins smartconnect_hp0/S02_AXI]

    # Connect the master interface of the SmartConnect to the PS High-Performance slave port
    connect_bd_intf_net [get_bd_intf_pins smartconnect_hp0/M00_AXI] $PS_S_AXI

    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master $PS_M_AXI_LITE Slave {/axi_cgra4ml_0/s_axil} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}]  [get_bd_intf_pins axi_cgra4ml_0/s_axil]
    
    # Connect the clock for the SmartConnect. The reset is inferred automatically.
    connect_bd_net [get_bd_pins smartconnect_hp0/aclk] [get_bd_pins processing_system7_0/FCLK_CLK0]

    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $PS_CLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
} else {
    set_property -dict [list CONFIG.AXIL_ADDR_WIDTH {40}] [get_bd_cells axi_cgra4ml_0]
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_output] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_pixel] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]
    connect_bd_intf_net [get_bd_intf_pins axi_cgra4ml_0/m_axi_weights] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD} Slave {/axi_cgra4ml_0/s_axil} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_cgra4ml_0/s_axil]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ultra_ps_e_0/pl_clk0 (250 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ultra_ps_e_0/pl_clk0 (250 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ultra_ps_e_0/pl_clk0 (250 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ultra_ps_e_0/saxihpc1_fpd_aclk]
}

generate_target all [get_files ./${PROJECT_NAME}/${PROJECT_NAME}.srcs/sources_1/bd/design_1/design_1.bd]
make_wrapper -files [get_files ./${PROJECT_NAME}/${PROJECT_NAME}.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./${PROJECT_NAME}/${PROJECT_NAME}.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Set AXl-lite and full_AXI addresses
if {$BOARD eq "pynq_z2"} {
    set_property range 256M [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_cgra4ml_0_reg0}]
    set_property offset ${CONFIG_BASEADDR} [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_cgra4ml_0_reg0}]
} else {
    set_property range 256M [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_axi_cgra4ml_0_reg0}]
    set_property offset ${CONFIG_BASEADDR} [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_axi_cgra4ml_0_reg0}]
}
assign_bd_address
save_bd_design

validate_bd_design
save_bd_design

# Implementation
reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 24
wait_on_run -timeout 360 impl_1
write_hw_platform -fixed -include_bit -force -file $PROJECT_NAME/design_1_wrapper.xsa

# Reports
open_run impl_1
if {![file exists $PROJECT_NAME/reports]} {exec mkdir $PROJECT_NAME/reports}
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 100 -input_pins -routable_nets -name timing_1 -file $PROJECT_NAME/reports/${PROJECT_NAME}_${BOARD}_${FREQ}_timing_report.txt
report_utilization -file $PROJECT_NAME/reports/${PROJECT_NAME}_${BOARD}_${FREQ}_utilization_report.txt -name utilization_1
report_power -file $PROJECT_NAME/reports/${PROJECT_NAME}_${BOARD}_${FREQ}_power_1.txt -name {power_1}
report_drc -name drc_1 -file $PROJECT_NAME/reports/${PROJECT_NAME}_${BOARD}_${FREQ}_drc_1.txt -ruledecks {default opt_checks placer_checks router_checks bitstream_checks incr_eco_checks eco_checks abs_checks}

exec mkdir -p $PROJECT_NAME/output
exec cp "$PROJECT_NAME/$PROJECT_NAME.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh" $PROJECT_NAME/output/
exec cp "$PROJECT_NAME/$PROJECT_NAME.runs/impl_1/design_1_wrapper.bit" $PROJECT_NAME/output/design_1.bit

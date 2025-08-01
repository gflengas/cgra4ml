# Set the Vitis workspace directory
setws /home/gflengas/cgra4ml/run/work/vitis

# --- Corresponds to GUI Step 1: Create Platform from .xsa ---
# Create the hardware platform from the Vivado-generated .xsa
platform create -name {dsf_pynq_z2_platform} -hw {/home/gflengas/cgra4ml/run/work/dsf_pynq_z2/design_1_wrapper.xsa}
platform active {dsf_pynq_z2_platform}

# Create the software environment (standalone OS) for the ARM core
domain create -name {standalone_ps7_cortexa9_0} -os {standalone} -proc {ps7_cortexa9_0}

# Build the platform, which generates the BSPs and bootloaders
platform generate

# --- Corresponds to GUI Step 1: Create Application Project ---
# Create the application using the "Hello World" template. This is crucial
# because it automatically generates the platform.h and platform.c files.
app create -name {dsf_app} -platform {dsf_pynq_z2_platform} -domain {standalone_ps7_cortexa9_0} -template {Hello World}

# --- This automates the manual step of replacing the template's code ---
# The original developer's script used 'cp' for this. We use the Tcl equivalent.
# This copies your xilinx_example.c over the template's helloworld.c.
set app_src_dir "/home/gflengas/cgra4ml/run/work/vitis/dsf_app/src"
set user_main_file "/home/gflengas/cgra4ml/deepsocflow/c/xilinx_example.c"
set template_main_file "$app_src_dir/helloworld.c"
file copy -force $user_main_file $template_main_file

# --- Corresponds to GUI Steps 2, 3, and 4: Configure Properties ---
# Add the math library (-lm)
app config -name {dsf_app} -add libraries {m}
# Set the compiler optimization level
app config -name {dsf_app} -set compiler-optimization {Optimize most (-O3)}
# Add the required include paths
app config -name {dsf_app} -add include-path "/home/gflengas/cgra4ml/deepsocflow/c"
app config -name {dsf_app} -add include-path "/home/gflengas/cgra4ml/run/work"

# --- Corresponds to GUI Step 5: Build ---
# Build the final application.
app build -name {dsf_app}
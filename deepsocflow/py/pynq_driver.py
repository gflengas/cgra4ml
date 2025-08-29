import pynq  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import json
import os

# --- Utility Functions (Ported from deepsocflow.py.utils and enhanced) ---
def pack_words_into_bytes(arr, bits):
    """Packs an array of integers (words) of specified bit-width into a byte array."""
    if bits == 8:
        return arr.astype(np.int8).tobytes()
    elif bits < 8:
        # Determine the correct dtype for intermediate calculations based on signedness
        signed = np.any(arr < 0)
        temp_dtype = np.int16 if signed else np.uint16
        words_per_byte = 8 // bits
        packed_size = (arr.size + words_per_byte - 1) // words_per_byte
        packed_bytes = np.zeros(packed_size, dtype=np.uint8)
        mask = (1 << bits) - 1

        for i in range(arr.size):
            byte_idx = i // words_per_byte
            bit_offset = (i % words_per_byte) * bits
            val = arr[i].astype(temp_dtype) & mask
            packed_bytes[byte_idx] |= (val << bit_offset)
        return packed_bytes.tobytes()

def unpack_bytes_into_words(byte_arr, bits):
    """
    Unpacks a byte array into an array of integers (words) of specified bit-width,
    correctly handling signed two's complement for arbitrary bit-widths.
    """
    if not isinstance(byte_arr, bytes):
        byte_arr = byte_arr.tobytes()

    if bits not in [4, 8, 16, 32]:
        raise ValueError(f"Unpacking for {bits}-bit words is not supported.")

    if bits == 8: return np.frombuffer(byte_arr, dtype=np.int8)
    if bits == 16: return np.frombuffer(byte_arr, dtype=np.int16)
    if bits == 32: return np.frombuffer(byte_arr, dtype=np.int32)
    
    # Custom logic for 4-bit unpacking
    if bits == 4:
        # Each byte contains two 4-bit words (nibbles)
        first_nibbles = (np.frombuffer(byte_arr, dtype=np.uint8) & 0x0F).astype(np.int8)
        second_nibbles = (np.frombuffer(byte_arr, dtype=np.uint8) >> 4).astype(np.int8)
        
        # Handle two's complement for negative numbers (sign bit is the 4th bit)
        first_nibbles[first_nibbles > 7] -= 16
        second_nibbles[second_nibbles > 7] -= 16

        # Interleave them back into the correct order
        unpacked_words = np.empty(len(byte_arr) * 2, dtype=np.int8)
        unpacked_words[0::2] = first_nibbles
        unpacked_words[1::2] = second_nibbles
        return unpacked_words

# --- End Utility Functions ---


class DeepSoCFlowPYNQ:
    """
    A PYNQ-based driver for the DeepSoCFlow accelerator.
    This version correctly pre-loads all bundle configurations into a single
    buffer, matching the C runtime's architecture.
    """

    def __init__(self, overlay: pynq.Overlay, config_path: str, accelerator_ip_name: str):
        if accelerator_ip_name not in overlay.ip_dict:
            raise AttributeError(f"Could not find IP '{accelerator_ip_name}' in overlay.ip_dict. "
                                 f"Available IPs are: {list(overlay.ip_dict.keys())}")
        
        ip_description = overlay.ip_dict[accelerator_ip_name]
        self.accelerator = pynq.overlay.DefaultIP(description=ip_description)
        self.mmio = self.accelerator.mmio
        
        print(f"\n--- Accelerator '{accelerator_ip_name}' Hardware Info ---")
        phys_addr = ip_description['phys_addr']
        addr_range = ip_description['addr_range']
        print(f"  > Physical Address Range: 0x{phys_addr:08x} - 0x{phys_addr + addr_range - 1:08x}")
        print(f"  > Address Range Size: {addr_range} bytes")
        print(f"--------------------------------------------------")

        print(f"\nLoading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.defines = config['defines']
        self.bundles = config['bundles']

        # Add the bundle index 'ib' to each bundle for easier debugging.
        for i, b in enumerate(self.bundles):
            b['ib'] = i

        self.mem = {}
        
        self._str_to_dtype = {
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
            'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
            'float32': np.float32, 'float64': np.float64
        }
        
        self.REG_OFFSETS = {
            'A_START'       : 0x0,
            'A_DONE_READ'   : 0x1,
            'A_DONE_WRITE'  : 0x3,
            'A_OCM_BASE'    : 0x5,
            'A_WEIGHTS_BASE': 0x7,
            'A_BUNDLE_DONE' : 0x8,
            'A_N_BUNDLES_1' : 0x9,
            'A_W_DONE'      : 0xA,
            'A_X_DONE'      : 0xB,
            'A_O_DONE'      : 0xC,
            'A_PARAMS_BASE' : 16, # Start offset for parameter BRAM, as per C-runtime
        }

        self._allocate_memory()

    def _allocate_memory(self):
        print("Allocating memory buffers...")
        defs = self.defines
        y_type = self._str_to_dtype[defs['Y_TYPE_str']]
        b_type = self._str_to_dtype[defs['B_TYPE_str']]
        o_type = self._str_to_dtype[defs['O_TYPE_str']]

        self.mem['ocm'] = pynq.allocate(shape=(2, defs['PE_COLS'] * defs['PE_ROWS']), dtype=y_type)
        self.mem['nhwc'] = pynq.allocate(shape=(defs['NHWC_WORDS'],), dtype=np.int32)
        self.mem['out_buffers'] = pynq.allocate(shape=(defs['N_OUT_BUF'], defs['O_BYTES_MAX']), dtype=np.int8)
        self.mem['w'] = pynq.allocate(shape=(defs['W_BYTES'],), dtype=np.int8)
        self.mem['b'] = pynq.allocate(shape=(defs['B_WORDS'],), dtype=b_type)
        self.mem['x'] = pynq.allocate(shape=(defs['X_BYTES'],), dtype=np.int8)
        self.mem['y'] = pynq.allocate(shape=(defs['O_WORDS'],), dtype=o_type)
        
        if defs['N_ADD_BUF'] > 0:
            self.mem['add_buffers'] = pynq.allocate(shape=(defs['N_ADD_BUF'], defs['NHWC_WORDS']), dtype=np.int8)
        
        # Per C-runtime, parameters are written to accelerator BRAM, but we still
        # allocate a buffer here for the notebook's verification steps.
        self.mem['params'] = pynq.allocate(shape=(defs['N_BUNDLES'], 8), dtype=np.uint32)

        print("Memory allocation complete.")

    def model_setup(self, wbx_path: str):
        print(f"\nSetting up model from {wbx_path}...")
        
        w_bytes = self.mem['w'].nbytes
        b_bytes = self.mem['b'].nbytes
        
        with open(wbx_path, 'rb') as f:
            wbx_data = f.read()

        np.copyto(self.mem['w'], np.frombuffer(wbx_data[:w_bytes], dtype=self.mem['w'].dtype))
        np.copyto(self.mem['b'], np.frombuffer(wbx_data[w_bytes : w_bytes + b_bytes], dtype=self.mem['b'].dtype))
        np.copyto(self.mem['x'], np.frombuffer(wbx_data[w_bytes + b_bytes:], dtype=self.mem['x'].dtype))
        
        self.mem['w'].flush()
        self.mem['b'].flush()
        self.mem['x'].flush()
        print("\nData copy complete.")

        print("Pre-loading all bundle parameters into accelerator BRAM...")
        
        # Use the allocated buffer from _allocate_memory
        params_buf = self.mem['params']

        for ib, b in enumerate(self.bundles):
            x_buf = self.mem['x'] if b['in_buffer_idx'] == -1 else self.mem['out_buffers'][b['in_buffer_idx']]
            
            # This parameter structure mimics the C-runtime
            params_buf[ib][0] = x_buf.physical_address
            params_buf[ib][1] = b['x_bpt_p0']
            params_buf[ib][2] = b['x_bpt']
            params_buf[ib][3] = b['w_bpt_p0']
            params_buf[ib][4] = b['w_bpt']
            params_buf[ib][5] = (b['t'] << 16) + b['p']
            
            header = b['header']
            params_buf[ib][6] = header & 0xFFFFFFFF
            params_buf[ib][7] = header >> 32
        
        # Write the generated parameters directly into the accelerator's register file / BRAM
        params_flat = params_buf.flatten()
        for i, val in enumerate(params_flat):
            self.mmio.write((self.REG_OFFSETS['A_PARAMS_BASE'] + i) * 4, int(val))

        print("Parameter loading complete.")

        print("--- DEBUG: Writing to HW Registers (C-Runtime Style) ---")
        self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_DONE_READ'] + 0) * 4, 1)
        self.mmio.write((self.REG_OFFSETS['A_DONE_READ'] + 1) * 4, 1)
        self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + 0) * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + 1) * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_OCM_BASE'] + 0) * 4, self.mem['ocm'][0].physical_address)
        self.mmio.write((self.REG_OFFSETS['A_OCM_BASE'] + 1) * 4, self.mem['ocm'][1].physical_address)
        self.mmio.write(self.REG_OFFSETS['A_WEIGHTS_BASE'] * 4, self.mem['w'].physical_address)
        self.mmio.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)
        self.mmio.write(self.REG_OFFSETS['A_N_BUNDLES_1'] * 4, self.defines['N_BUNDLES'])
        # Initialize the status registers that were missing (critical!)
        self.mmio.write(self.REG_OFFSETS['A_W_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_X_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_O_DONE'] * 4, 0)
        print("Register configuration complete.")
        print("Model setup finished.")

    def model_run(self, input_data=None):
        print("\n--- Starting Model Run ---")
        # Debug: Check if input data is loaded correctly
        print(f"DEBUG: Input data (first 16 bytes): {self.mem['x'][:16]}")
        print(f"DEBUG: Weight data (first 16 bytes): {self.mem['w'][:16]}")
        print(f"DEBUG: Bias data (first 8 values): {self.mem['b'][:8]}")

        if input_data is not None:
            np.copyto(self.mem['x'], np.frombuffer(input_data, dtype=np.int8))
            self.mem['x'].flush()

        config_base = self.mmio
        ocm_bank = 1 # Will be flipped to 0 on first iteration

        # Check initial state before starting
        print("--- Initial Hardware State ---")
        for bank in [0, 1]:
            done_write = config_base.read((self.REG_OFFSETS['A_DONE_WRITE'] + bank) * 4)
            done_read = config_base.read((self.REG_OFFSETS['A_DONE_READ'] + bank) * 4)
            print(f"Bank {bank}: DONE_WRITE={done_write}, DONE_READ={done_read}")
        
        w_done = config_base.read(self.REG_OFFSETS['A_W_DONE'] * 4)
        x_done = config_base.read(self.REG_OFFSETS['A_X_DONE'] * 4)
        o_done = config_base.read(self.REG_OFFSETS['A_O_DONE'] * 4)
        start_val = config_base.read(self.REG_OFFSETS['A_START'] * 4)
        print(f"Status: START={start_val}, W_DONE={w_done}, X_DONE={x_done}, O_DONE={o_done}")

        # Start the accelerator ONCE at the beginning (like C runtime)
        config_base.write(self.REG_OFFSETS['A_START'] * 4, 1)
        print("Set A_START=1")
        
        # Give hardware a moment to initialize
        import time
        time.sleep(0.1)
        
        # Check state after start
        print("--- State After A_START=1 ---")
        for bank in [0, 1]:
            done_write = config_base.read((self.REG_OFFSETS['A_DONE_WRITE'] + bank) * 4)
            done_read = config_base.read((self.REG_OFFSETS['A_DONE_READ'] + bank) * 4)
            print(f"Bank {bank}: DONE_WRITE={done_write}, DONE_READ={done_read}")
        
        w_done = config_base.read(self.REG_OFFSETS['A_W_DONE'] * 4)
        x_done = config_base.read(self.REG_OFFSETS['A_X_DONE'] * 4)
        o_done = config_base.read(self.REG_OFFSETS['A_O_DONE'] * 4)
        print(f"Status: W_DONE={w_done}, X_DONE={x_done}, O_DONE={o_done}")

        for ib, b in enumerate(self.bundles):
            print(f"\n--- Processing Bundle {ib} ---")
            
            # Add detailed bundle configuration for all bundles
            if ib in [0, 1, 2, 3, 4, 5, 6]:
                print(f"Bundle {ib} Config: p={b['p']}, t={b['t']}, n={b['n']}, l={b['l']}, w_kw2={b['w_kw2']}")
                print(f"Bundle {ib} Dims: h={b['h']}, w={b['w']}, co={b['co']}, coe={b['coe']}")
                print(f"Bundle {ib} Bias: is_bias={b['is_bias']}, b_offset={b['b_offset']}")
            
            is_last_bundle = (ib == len(self.bundles) - 1)
            o_buf = self.mem['y'] if is_last_bundle else self.mem['out_buffers'][b['out_buffer_idx']]

            # Allocate separate buffers for pre-stride accumulation and post-stride results
            nhwc_buf = np.zeros(b['n'] * b['ch'] * b['cw'] * b['co'], dtype=np.int32)
            p_pass_buf = np.zeros(b['n'] * b['h'] * b['w'] * b['co'], dtype=np.int32) if b['p'] > 1 else None

            for p in range(b['p']):
                for t in range(b['t']):
                    for n in range(b['n']):
                        for l in range(b['l']):
                            for w_kw2 in range(b['w_kw2']):
                                # print(f"  > B{ib} P{p} T{t} N{n} L{l} W_KW2:{w_kw2} | Bank: {1-ocm_bank}")
                                ocm_bank = 1 - ocm_bank # Flip bank
                                
                                # Check current state
                                done_write = config_base.read((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4)
                                # print(f"    About to wait for DONE_WRITE on bank {ocm_bank}")
                                # print(f"    Current DONE_WRITE[{ocm_bank}] = {done_write}")
                                
                                if done_write == 1:
                                    # print(f"    DONE_WRITE[{ocm_bank}] is already 1! Proceeding immediately.")
                                    pass
                                else:
                                    # Wait for the accelerator to finish writing to the current OCM bank
                                    # print(f"    ... Waiting for accelerator to write to bank {ocm_bank}")
                                    timeout_counter = 0
                                    while config_base.read((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4) == 0:
                                        timeout_counter += 1
                                        if timeout_counter % 10 == 0:  # Print debug every 10 iterations
                                            w_done = config_base.read(self.REG_OFFSETS['A_W_DONE'] * 4)
                                            x_done = config_base.read(self.REG_OFFSETS['A_X_DONE'] * 4)
                                            o_done = config_base.read(self.REG_OFFSETS['A_O_DONE'] * 4)
                                            print(f"      - Still waiting... Internal states: W_DONE={w_done}, X_DONE={x_done}, O_DONE={o_done}")
                                        time.sleep(0.01)  # Shorter sleep for faster polling
                                        if timeout_counter > 1000:  # Timeout after 10 seconds
                                            print(f"ERROR: Timeout waiting for DONE_WRITE on bank {ocm_bank}")
                                            break
                                    # print(f"    ... Accelerator finished writing to bank {ocm_bank}")
                                
                                # Clear the DONE_WRITE flag immediately (matching C runtime)
                                config_base.write((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4, 0)
                                # print(f"    Cleared DONE_WRITE[{ocm_bank}]")
                                
                                # Process the data from OCM
                                y_tile_ocm = self.mem['ocm'][ocm_bank]
                                y_tile_ocm.invalidate()
                                
                                w_last = (b['kw'] // 2 + 1) if (w_kw2 == b['w_kw2'] - 1) else 1
                                self.process_tile_py(y_tile_ocm, nhwc_buf, p_pass_buf, b, l, w_kw2, w_last, n, p, t)
                                
                                # Signal to hardware that the CPU is done reading from this OCM bank
                                config_base.write((self.REG_OFFSETS['A_DONE_READ'] + ocm_bank) * 4, 1)
                                # print(f"    ... Signaled CPU done reading bank {ocm_bank}")

            # After all passes and tiles, perform pooling/packing on the assembled buffer
            self._perform_pooling_and_packing(nhwc_buf, o_buf, b)
            
            # Debug Bundle outputs AFTER processing is complete for ALL bundles
            print(f"\n--- Bundle {ib} Output Debug ---")
            if is_last_bundle:
                print(f"Bundle {ib} FINAL output (first 10): {self.mem['y'][:10]}")
            else:
                # Unpack the output buffer to see what the next bundle will receive as input
                x_bits = 1 << self.defines['X_BITS_L2']
                packed_data = self.mem['out_buffers'][b['out_buffer_idx']][:50]  # First 50 bytes
                unpacked = unpack_bytes_into_words(packed_data, x_bits)
                print(f"Bundle {ib} output (first 20): {unpacked[:20]}")
                
                # Also show NHWC buffer before packing for debugging
                print(f"Bundle {ib} NHWC buffer (first 20): {nhwc_buf[:20]}")
            
            # Signal that the entire bundle is done
            config_base.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)

        print("\n--- Model Run Finished ---")
        return self._get_final_output()

    def process_tile_py(self, y_tile_in, nhwc_buf, p_pass_buf, b, il, iw_kw2, w_last, n, p, t):
        """
        Rewritten to exactly match the C runtime logic in runtime.h
        """
        # --- DEBUG PRINT: Verify raw OCM input to CPU ---
        if b['ib'] == 0 and p == 0 and t == 0 and n == 0 and il == 0 and iw_kw2 == 0:
            print(f"    DEBUG Bundle 0 First Tile: Raw OCM data: {y_tile_in[:16]}")
        
        # Add debugging for Bundle 6 (final bundle)
        if b['ib'] == 6:
            print(f"    DEBUG B{b['ib']}: P{p} T{t} N{n} L{il} W_KW2{iw_kw2}")
            print(f"    DEBUG B{b['ib']}: Raw OCM data: {y_tile_in[:16]}")
        
        # Add debugging for Bundle 5 (input to Bundle 6)
        if b['ib'] == 5:
            print(f"    DEBUG B{b['ib']}: P{p} T{t} N{n} L{il} W_KW2{iw_kw2}")
            print(f"    DEBUG B{b['ib']}: Raw OCM data: {y_tile_in[:16]}")
        
        pe_rows = self.defines['PE_ROWS']
        x_bits_l2 = self.defines['X_BITS_L2']
        x_bits = 1 << x_bits_l2

        sram_addr = 0
        processed_values = []
        
        # Detailed debugging for Bundle 0 first few values
        detailed_debug = (b['ib'] == 0 and p == 0 and t == 0 and n == 0 and il == 0 and iw_kw2 == 0)
        
        for icoe in range(b['coe']):
            i_bias = b['b_offset'] + b['coe'] * t + icoe
            
            for iw_last in range(w_last):
                for ir in range(pe_rows):
                    # Calculate indices exactly like C runtime
                    i_yn = n
                    i_yh = il * pe_rows + ir
                    i_yw = iw_kw2 + iw_last
                    i_yc = b['coe'] * t + icoe
                    
                    # Save dimensions exactly like C runtime
                    yn = b['n']
                    yh = b['h'] 
                    yw = b['w']
                    yc = b['co']
                    
                    # DETAILED DEBUGGING FOR FIRST FEW VALUES OF BUNDLE 0
                    if detailed_debug and sram_addr < 8:
                        print(f"      === Processing sram_addr={sram_addr}, i_yh={i_yh}, i_yw={i_yw}, i_yc={i_yc} ===")
                    
                    # If out of bounds, early return (matching C runtime)
                    if i_yh >= yh or i_yc >= yc:
                        if detailed_debug and sram_addr < 8:
                            print(f"      OUT OF BOUNDS: i_yh={i_yh}>={yh} or i_yc={i_yc}>={yc}")
                        if p == b['p'] - 1:
                            pass  # C runtime: sim_fprintf for last p
                        sram_addr += 1
                        continue

                    raw_val = y_tile_in[sram_addr]
                    out_val = raw_val.astype(np.int64)
                    
                    if detailed_debug and sram_addr < 8:
                        print(f"      raw_val={raw_val}, out_val={out_val}")
                    
                    sram_addr += 1

                    # ------ ADD P PASSES ------ (exactly like C runtime)
                    iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc)
                    
                    if b['p'] == 1:
                        # only p: proceed with value
                        pass
                    elif p == b['p'] - 1:
                        # last p: read, add, proceed
                        out_val += p_pass_buf[iy_nhwc]
                    elif p == 0:
                        # first p: overwrite memory, return
                        p_pass_buf[iy_nhwc] = out_val
                        if detailed_debug and sram_addr <= 8:
                            print(f"      FIRST P: stored {out_val} at nhwc_idx={iy_nhwc}")
                        continue
                    else:
                        # middle p: read, add, store, return
                        p_pass_buf[iy_nhwc] += out_val
                        if detailed_debug and sram_addr <= 8:
                            print(f"      MIDDLE P: added {out_val} at nhwc_idx={iy_nhwc}")
                        continue

                    if detailed_debug and sram_addr <= 8:
                        print(f"      After P passes: out_val={out_val}")

                    # ------ CONV STRIDING ------ (exactly like C runtime)
                    if (i_yh - b['csh_shift']) % b['csh'] != 0 or (i_yw - b['csw_shift']) % b['csw'] != 0:
                        if detailed_debug and sram_addr <= 8:
                            print(f"      CONV STRIDING SKIP")
                        continue

                    i_yh = (i_yh - b['csh_shift']) // b['csh']  # update indices like C runtime
                    i_yw = (i_yw - b['csw_shift']) // b['csw']
                    yh = b['ch']  # update dimensions like C runtime
                    yw = b['cw']

                    if detailed_debug and sram_addr <= 8:
                        print(f"      After striding: i_yh={i_yh}, i_yw={i_yw}, yh={yh}, yw={yw}")

                    # ------ ADD BIAS ------ (exactly like C runtime)
                    if b['is_bias']:
                        out_val = (out_val << b['b_val_shift']) + (self.mem['b'][i_bias].astype(np.int64) << b['b_bias_shift'])
                        if detailed_debug and sram_addr <= 8:
                            print(f"      After bias: out_val={out_val}")

                    # ------ CORE ACT ------ (exactly like C runtime)
                    out_val = self._quant_lrelu(out_val, b['ca_nzero'], b['ca_shift'], b['ca_pl_scale'])
                    
                    if detailed_debug and sram_addr <= 8:
                        print(f"      After core activation: out_val={out_val}")

                    # ------ RESIDUAL ADD --- (exactly like C runtime)
                    if b['add_in_buffer_idx'] != -1:
                        iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc)
                        out_val += self.mem['add_buffers'][b['add_in_buffer_idx']][iy_nhwc]
                        out_val = self._quant_lrelu(out_val, b['aa_nzero'], b['aa_shift'], b['aa_pl_scale'])

                    # ------ SOFTMAX ------ (skip for now, handled elsewhere)
                    if b.get('is_softmax', False):
                        # Handle softmax later in the pipeline
                        pass

                    # ------ MAX/AVG POOL --- (exactly like C runtime logic)
                    if b.get('pool', 'POOL_NONE') == 'POOL_NONE':
                        # NO POOLING: Call tile_write equivalent
                        self._tile_write_py(out_val, b, i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc, nhwc_buf)
                        if detailed_debug and sram_addr <= 8:
                            print(f"      NO POOLING: called tile_write_py")
                        continue
                    else:
                        # POOLING: Store in NHWC buffer for pooling (existing logic)
                        iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc)
                        nhwc_buf[iy_nhwc] = out_val
                        # TODO: Implement pooling logic here if needed
                    
                    # Debug: collect first few processed values from Bundle 0
                    if b['ib'] == 0 and len(processed_values) < 8:
                        processed_values.append((raw_val, out_val, i_yh, i_yw, i_yc))
        
        # Debug print for Bundle 0
        if b['ib'] == 0 and p == 0 and t == 0 and n == 0 and il == 0 and iw_kw2 == 0 and processed_values:
            print(f"    DEBUG Bundle 0 Processing Summary:")
            for i, (raw, final, yh, yw, yc) in enumerate(processed_values[:4]):
                print(f"      [{i}] raw={raw} -> final={final} (yh={yh}, yw={yw}, yc={yc})")

    def _flatten_nhwc(self, i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc):
        """
        Exactly matches the C runtime flatten_nhwc macro
        """
        return ((i_yn * yh + i_yh) * yw + i_yw) * yc + i_yc

    def _tile_write_py(self, out_val, b, i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc, nhwc_buf):
        """
        Python equivalent of the C runtime tile_write function
        This is the critical function we were missing!
        """
        # ------ FLATTEN ------ (exactly like C runtime)
        if b.get('is_flatten', False):
            i_yc = (i_yh * yw + i_yw) * yc + i_yc  # (H*W*C) -> C
            i_yw = 0                               # W=1
            i_yh = i_yn                           # N -> H
            i_yn = 0                              # N=1
            
            yc = yh * yw * yc
            yw = 1
            yh = yn
            yn = 1

        # ------ STORE IN NHWC ------ (exactly like C runtime)
        iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, b['on'], b['oh'], b['ow'], b['oc'])
        
        # For debugging, also store in our nhwc_buf (equivalent to mp->debug_nhwc)
        if iy_nhwc < len(nhwc_buf):
            nhwc_buf[iy_nhwc] = out_val
        
        is_last_bundle = (b['ib'] == len(self.bundles) - 1)
        
        if is_last_bundle:
            # Last bundle: save as NHWC in final output
            if iy_nhwc < len(self.mem['y']):
                self.mem['y'][iy_nhwc] = out_val
            return

        # Store for residual add (if needed)
        if b.get('add_out_buffer_idx', -1) != -1:
            if iy_nhwc < len(self.mem['add_buffers'][b['add_out_buffer_idx']]):
                self.mem['add_buffers'][b['add_out_buffer_idx']][iy_nhwc] = np.int8(out_val)

        # If output only goes to residual add, early return
        if b.get('ib_out', -1) == -1:
            return

        # ------ TILING: Calculate X coordinates ------ (complex tiling logic)
        # For now, we'll use the simplified approach since the tiling logic is very complex
        # TODO: Implement full tiling logic if this doesn't work
        
        # For Bundle 0, let's check if this is supposed to be packed
        if not is_last_bundle:
            # Pack the data into the output buffer (simplified version)
            o_buf = self.mem['out_buffers'][b['out_buffer_idx']]
            x_bits = 1 << self.defines['X_BITS_L2']
            
            # For now, use simple indexing - we may need to implement full tiling later
            if iy_nhwc < len(nhwc_buf):
                nhwc_buf[iy_nhwc] = out_val

    def _quant_lrelu(self, x, nzero, shift, pl_scale):
        """
        Exactly matches the C runtime quant_lrelu function
        """
        x_bits = 1 << self.defines['X_BITS_L2']
        
        # Conditional, targeting ARM (exactly like C runtime)
        x = x if (x < 0 and nzero) or (x >= 0) else 0
        if x >= 0:
            x = x << pl_scale
        x = self.shift_round(x, shift)
        x = np.clip(x, -(1 << (x_bits - pl_scale - 1)), (1 << (x_bits - 1)) - 1)
        return x

    def shift_round(self, n, s):
        """
        Implements the exact C runtime shift_round behavior:
        shift_round(n, s) = (((n) + ((s)>0 ? (1<<((s)-1)) - (~((n)>>(s))&1) : 0)) >> s)
        """
        if s <= 0:
            return n >> s if s < 0 else n
        
        # Calculate the rounding adjustment
        round_adjust = (1 << (s - 1)) - (~((n >> s) & 1) & 1)
        return (n + round_adjust) >> s

    def div_round(self, a, b):
        """
        Implements the exact C runtime div_round behavior:
        div_round(a, b) = (((a)+((b)/2) - (~((b)|(a)/(b)) &1))/(b))
        """
        return ((a + (b // 2) - (~((b | (a // b)) & 1) & 1)) // b)


    def _perform_pooling_and_packing(self, nhwc_buf, o_buf, b):
        """
        Performs the final pooling, flattening, and packing operations on the
        fully assembled NHWC buffer for a bundle.
        """
        if b.get('is_flatten', False):
            valid_data_size = b['n'] * b['ch'] * b['cw'] * b['co']
            processed_data = nhwc_buf[:valid_data_size]
        else:
            # Reshape based on post-stride dimensions, which are now correctly
            # stored in the dense nhwc_buf.
            valid_data_size = b['n'] * b['ch'] * b['cw'] * b['co']
            img = nhwc_buf[:valid_data_size].reshape(b['n'], b['ch'], b['cw'], b['co'])

            if b['pool'] != 'POOL_NONE':
                pooled_rows = b['oh']
                pooled_cols = b['ow']
                pooled_img = np.zeros((b['n'], pooled_rows, pooled_cols, b['co']), dtype=np.int32)
                
                for r in range(pooled_rows):
                    for c in range(pooled_cols):
                        r_start, c_start = r * b['psh'], c * b['psw']
                        r_end, c_end = r_start + b['pkh'], c_start + b['pkw']
                        window = img[:, r_start:r_end, c_start:c_end, :]
                        
                        if b['pool'] == 'POOL_MAX':
                            pooled_img[:, r, c, :] = np.max(window, axis=(1, 2))
                        elif b['pool'] == 'POOL_AVG':
                            # Calculate sum first, then use div_round like C runtime
                            window_sum = np.sum(window, axis=(1, 2))
                            count = window.shape[1] * window.shape[2]  # window size
                            avg_val = np.array([self.div_round(int(s), count) for s in window_sum.flatten()])
                            avg_val = avg_val.reshape(window_sum.shape)
                            
                            # Apply activation function like C runtime does
                            x_bits = 1 << self.defines['X_BITS_L2']
                            activated_val = np.array([
                                self._quant_lrelu(int(val), b['pa_nzero'], b['pa_shift'], b['pa_pl_scale'])
                                for val in avg_val.flatten()
                            ]).reshape(avg_val.shape)
                            
                            pooled_img[:, r, c, :] = activated_val
                
                processed_data = pooled_img
            else:
                processed_data = img

        # --- Flatten and Pack ---
        output_words = processed_data.flatten()
        
        is_last_bundle = (b['ib'] == len(self.bundles) - 1)

        if is_last_bundle:
            # For the final output, we do not pack. The data is int32.
            # o_buf is self.mem['y'] which is already of the correct dtype.
            np.copyto(o_buf[:output_words.size], output_words)
        else:
            # For intermediate layers, pack the data to the network's internal bit-width.
            x_bits = 1 << self.defines['X_BITS_L2']
            packed_bytes = pack_words_into_bytes(output_words, x_bits)
            
            # Create a view of the packed bytes with the correct dtype of the output buffer
            packed_as_dtype = np.frombuffer(packed_bytes, dtype=o_buf.dtype)
            
            # Copy only the generated data into the beginning of the output buffer slice
            np.copyto(o_buf[:packed_as_dtype.size], packed_as_dtype)

        o_buf.flush()

    def _get_final_output(self):
        """
        Reads the final output buffer, applies softmax if needed, and returns the result.
        """
        last_bundle = self.bundles[-1]
        
        # The final output buffer self.mem['y'] contains the raw integer words.
        final_output_words = self.mem['y']
        
        # Apply softmax if this is the last layer
        if last_bundle['is_softmax']:
            # De-quantize the output words according to C-runtime logic
            softmax_frac = last_bundle['softmax_frac']
            softmax_max_f = last_bundle['softmax_max_f']
            
            # Perform operations on a float copy
            float_words = final_output_words.astype(np.float32)
            float_words /= (1 << softmax_frac)
            float_words -= softmax_max_f
            
            # Reshape to apply softmax along the channel axis, mimicking the C-runtime.
            # The C-runtime applies softmax per-pixel, over the channel dimension.
            num_classes = last_bundle['co']
            if num_classes == 0:
                final_output = float_words # Avoid division by zero
            else:
                num_vectors = last_bundle['o_words'] // num_classes
                
                if num_vectors * num_classes != last_bundle['o_words']:
                    # Fallback for unexpected shapes, though this indicates a config issue.
                    exp_values = np.exp(float_words)
                    sum_exp_values = np.sum(exp_values)
                    final_output = exp_values / sum_exp_values if sum_exp_values != 0 else exp_values
                else:
                    valid_words = float_words[:last_bundle['o_words']]
                    reshaped_words = valid_words.reshape((num_vectors, num_classes))

                    exp_values = np.exp(reshaped_words)
                    sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)

                    # Avoid division by zero with a more robust approach
                    # Set a minimum threshold to avoid numerical issues
                    sum_exp_values = np.maximum(sum_exp_values, 1e-10)
                    final_output_reshaped = exp_values / sum_exp_values
                    final_output = final_output_reshaped.flatten()
        else:
            final_output = final_output_words

        # Return only the valid part of the output buffer
        return final_output[:last_bundle['o_words']]
        
    def __del__(self):
        print("\nReleasing memory buffers.")
        for name, buf in self.mem.items():
            if hasattr(buf, 'freebuffer'):
                buf.freebuffer()

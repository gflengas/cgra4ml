import pynq  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import json
import os
import time

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
        
        # Calculate the total size of the Memory_st structure (in bytes)
        # Based on the C structure layout:
        y_type = self._str_to_dtype[defs['Y_TYPE_str']]
        b_type = self._str_to_dtype[defs['B_TYPE_str']]
        o_type = self._str_to_dtype[defs['O_TYPE_str']]
        
        # Calculate sizes in bytes for each component
        ocm_size = 2 * defs['PE_COLS'] * defs['PE_ROWS'] * np.dtype(y_type).itemsize
        nhwc_size = defs['NHWC_WORDS'] * 4  # int32
        out_buffers_size = defs['N_OUT_BUF'] * defs['O_BYTES_MAX']
        w_size = defs['W_BYTES']
        b_size = defs['B_WORDS'] * np.dtype(b_type).itemsize
        x_size = defs['X_BYTES']
        y_size = defs['O_WORDS'] * np.dtype(o_type).itemsize
        add_buffers_size = defs['N_ADD_BUF'] * defs['NHWC_WORDS'] if defs['N_ADD_BUF'] > 0 else 0
        
        # Calculate total size with alignment padding
        total_size = (ocm_size + nhwc_size + out_buffers_size + 
                    w_size + b_size + x_size + y_size + add_buffers_size)
        
        # Add some padding for alignment (round up to 4KB boundary)
        total_size = ((total_size + 4095) // 4096) * 4096
        
        print(f"Allocating single contiguous buffer of {total_size} bytes...")
        
        # Allocate one large contiguous buffer (as uint8 for byte-level access)
        self.mem_base = pynq.allocate(shape=(total_size,), dtype=np.uint8, cacheable=False)
        
        print(f"Memory base physical address: 0x{self.mem_base.physical_address:08x}")
        
        # Store physical addresses separately
        self.physical_addresses = {}
        
        # Create views into this buffer at the correct offsets (matching C struct layout)
        offset = 0
        
        # OCM banks [2][PE_COLS*PE_ROWS] - Y_TYPE (typically int16)
        ocm_elements_per_bank = defs['PE_COLS'] * defs['PE_ROWS']
        self.mem['ocm'] = []
        for i in range(2):
            ocm_view = self.mem_base[offset:offset + ocm_elements_per_bank * np.dtype(y_type).itemsize]
            self.mem['ocm'].append(ocm_view.view(dtype=y_type).reshape(ocm_elements_per_bank))
            self.physical_addresses[f'ocm_{i}'] = self.mem_base.physical_address + offset
            offset += ocm_elements_per_bank * np.dtype(y_type).itemsize
            print(f"OCM Bank {i} physical address: 0x{self.physical_addresses[f'ocm_{i}']:08x}")
        
        # NHWC buffer [NHWC_WORDS] - int32
        nhwc_view = self.mem_base[offset:offset + defs['NHWC_WORDS'] * 4]
        self.mem['nhwc'] = nhwc_view.view(dtype=np.int32).reshape(defs['NHWC_WORDS'])
        self.physical_addresses['nhwc'] = self.mem_base.physical_address + offset
        offset += defs['NHWC_WORDS'] * 4
        
        # Out buffers [N_OUT_BUF][O_BYTES_MAX] - int8
        out_buffers_view = self.mem_base[offset:offset + defs['N_OUT_BUF'] * defs['O_BYTES_MAX']]
        self.mem['out_buffers'] = out_buffers_view.view(dtype=np.int8).reshape(defs['N_OUT_BUF'], defs['O_BYTES_MAX'])
        self.physical_addresses['out_buffers'] = self.mem_base.physical_address + offset
        offset += defs['N_OUT_BUF'] * defs['O_BYTES_MAX']
        
        # Weights [W_BYTES] - int8
        w_view = self.mem_base[offset:offset + defs['W_BYTES']]
        self.mem['w'] = w_view.view(dtype=np.int8).reshape(defs['W_BYTES'])
        self.physical_addresses['w'] = self.mem_base.physical_address + offset  # Store physical address separately
        offset += defs['W_BYTES']
        
        # Biases [B_WORDS] - B_TYPE
        b_view = self.mem_base[offset:offset + defs['B_WORDS'] * np.dtype(b_type).itemsize]
        self.mem['b'] = b_view.view(dtype=b_type).reshape(defs['B_WORDS'])
        self.physical_addresses['b'] = self.mem_base.physical_address + offset
        offset += defs['B_WORDS'] * np.dtype(b_type).itemsize
        
        # Input [X_BYTES] - int8
        x_view = self.mem_base[offset:offset + defs['X_BYTES']]
        self.mem['x'] = x_view.view(dtype=np.int8).reshape(defs['X_BYTES'])
        self.physical_addresses['x'] = self.mem_base.physical_address + offset
        offset += defs['X_BYTES']
        
        # Output [O_WORDS] - O_TYPE
        y_view = self.mem_base[offset:offset + defs['O_WORDS'] * np.dtype(o_type).itemsize]
        self.mem['y'] = y_view.view(dtype=o_type).reshape(defs['O_WORDS'])
        self.physical_addresses['y'] = self.mem_base.physical_address + offset
        offset += defs['O_WORDS'] * np.dtype(o_type).itemsize
        
        # Add buffers (if any)
        if defs['N_ADD_BUF'] > 0:
            add_buffers_view = self.mem_base[offset:offset + defs['N_ADD_BUF'] * defs['NHWC_WORDS']]
            self.mem['add_buffers'] = add_buffers_view.view(dtype=np.int8).reshape(defs['N_ADD_BUF'], defs['NHWC_WORDS'])
            self.physical_addresses['add_buffers'] = self.mem_base.physical_address + offset
            offset += defs['N_ADD_BUF'] * defs['NHWC_WORDS']
        
        # Allocate parameters buffer separately (this goes to BRAM, not main memory)
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
        
        self.mem_base.flush()  # Flush the entire contiguous buffer
        print("\nData copy complete.")

        print("Pre-loading all bundle parameters into accelerator BRAM...")
        
        # Use the allocated buffer from _allocate_memory
        params_buf = self.mem['params']

        for ib, b in enumerate(self.bundles):
            # Use stored physical addresses instead of trying to access .physical_address on views
            if b['in_buffer_idx'] == -1:
                x_addr = self.physical_addresses['x']
            else:
                x_addr = self.physical_addresses['out_buffers'] + b['in_buffer_idx'] * self.defines['O_BYTES_MAX']
            
            # This parameter structure mimics the C-runtime
            params_buf[ib][0] = x_addr
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

        # Use stored physical addresses for hardware registers
        self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_DONE_READ'] + 0) * 4, 1)
        self.mmio.write((self.REG_OFFSETS['A_DONE_READ'] + 1) * 4, 1)
        self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + 0) * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + 1) * 4, 0)
        self.mmio.write((self.REG_OFFSETS['A_OCM_BASE'] + 0) * 4, self.physical_addresses['ocm_0'])
        self.mmio.write((self.REG_OFFSETS['A_OCM_BASE'] + 1) * 4, self.physical_addresses['ocm_1'])
        self.mmio.write(self.REG_OFFSETS['A_WEIGHTS_BASE'] * 4, self.physical_addresses['w'])
        self.mmio.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)
        self.mmio.write(self.REG_OFFSETS['A_N_BUNDLES_1'] * 4, self.defines['N_BUNDLES'])
        self.mmio.write(self.REG_OFFSETS['A_W_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_X_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_O_DONE'] * 4, 0)
        
        print("Register configuration complete.")
        print(f"OCM Bank 0 address: 0x{self.physical_addresses['ocm_0']:08x}")
        print(f"OCM Bank 1 address: 0x{self.physical_addresses['ocm_1']:08x}")
        print(f"Weights address: 0x{self.physical_addresses['w']:08x}") 
        print("Model setup finished.")


    def model_run(self):
        """
        Executes the model inference on the accelerator, mimicking the C-runtime.
        """

        # --- DEBUG: Print buffer contents before starting ---
        print("\n--- Verifying buffer contents at start of model_run ---")
        try:
            # 1. Verify 'w' (weights)
            w_bits = 1 << self.defines['W_BITS_L2']
            w_unpacked = unpack_bytes_into_words(self.mem['w'].tobytes(), w_bits)
            print("First 16 values in 'w' buffer:", w_unpacked[:16])
            
            # 2. Verify 'b' (biases)
            print("First 16 values in 'b' buffer:", self.mem['b'][:16])

            # 3. Verify 'x' (input)
            x_bits = 1 << self.defines['X_BITS_L2']
            x_unpacked = unpack_bytes_into_words(self.mem['x'].tobytes(), x_bits)
            print("First 16 values in 'x' buffer:", x_unpacked[:16])
        except Exception as e:
            print(f"!!! Error printing debug buffer contents: {e}")
        print("-------------------------------------------------------")


        print("\n--- Starting Model Inference ---")
        start_time = time.time()

        # Start the accelerator by holding A_START high for the duration of the run
        self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 1)

        ocm_bank = 1  # Will be flipped to 0 at the start of the first loop
        first_pixel_printed = False # Add a flag to print only once per run

        for ib, b in enumerate(self.bundles):
            print(f"Executing Bundle {ib}/{len(self.bundles)-1}...")
            
            # --- DEBUG: Print the input buffer for this bundle ---
            in_buffer_idx = b.get('in_buffer_idx', -1)
            if in_buffer_idx == -1:
                input_buf = self.mem['x']
                print("  > Using initial model input ('x' buffer).")
            else:
                input_buf = self.mem['out_buffers'][in_buffer_idx]
                print(f"  > Using output buffer {in_buffer_idx} from a previous bundle as input.")
            
            x_bits = 1 << self.defines['X_BITS_L2']
            unpacked_input = unpack_bytes_into_words(input_buf.tobytes(), x_bits)
            print(f"  > Input data (first 16 values): {unpacked_input[:16]}")
            # --- End DEBUG Print ---
            
            # This buffer will hold the fully assembled NHWC output for this layer
            nhwc_buf_shape = (b['n'], b['ch'], b['cw'], b['co'])
            nhwc_buf_size = np.prod(nhwc_buf_shape)
            nhwc_buf = np.zeros(nhwc_buf_size, dtype=np.int32)
            
            p_out_buffer = self.mem['y'] if ib == len(self.bundles) - 1 else self.mem['out_buffers'][b['out_buffer_idx']]

            for ip in range(b['p']):
                for it in range(b['t']):
                    it_bias = b['b_offset'] + b['coe'] * it

                    for in_ in range(b['n']):
                        for il in range(b['l']):
                            for iw_kw2 in range(b['w_kw2']):
                                ocm_bank = 1 - ocm_bank
                                w_last = b['kw'] // 2 + 1 if iw_kw2 == b['w_kw2'] - 1 else 1

                                # Calculate o_bpt like C runtime (for understanding data size)
                                o_bpt = self.defines['PE_ROWS'] * b['coe'] * w_last * 4  # sizeof(int32) = 4

                                # --- Wait for Accelerator ---
                                while not self.mmio.read((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4):
                                    pass # Busy-wait like the C-runtime
                               
                                # However, we can use o_bpt to understand how many valid elements we have
                                time.sleep(0.001) 
                                valid_elements = o_bpt // 4  # Convert bytes to int32 elements
                                # Invalidate the entire base buffer to ensure cache coherency
                                self.mem['ocm'][ocm_bank].sync_from_device()
                                self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4, 0)
                                if iw_kw2 == 0 and it == 0:
                                    print(f"\n--- Reading OCM Bank {ocm_bank} for Bundle {ib} (ip={ip}, it={it}, iw_kw2={iw_kw2}) ---")
                                    print(f"o_bpt: {o_bpt} bytes, valid_elements: {valid_elements}")
                                    print(f"OCM Bank {ocm_bank}: ", np.int32(self.mem['ocm'][ocm_bank][:min(32, valid_elements)]))

                                
                                # --- Process OCM Data (Python side) ---
                                sram_addr = 0
                                for icoe in range(b['coe']):
                                    i_bias = it_bias + icoe
                                    for iw_last in range(w_last):
                                        for ir in range(self.defines['PE_ROWS']):
                                            i_yn = in_
                                            i_yh = il * self.defines['PE_ROWS'] + ir
                                            i_yw = iw_kw2 + iw_last
                                            i_yc = b['coe'] * it + icoe
                                            
                                            yn, yh, yw, yc = b['n'], b['h'], b['w'], b['co']

                                            if i_yh >= yh or i_yc >= yc:
                                                sram_addr += 1
                                                continue
                                            
                                            raw_val = self.mem['ocm'][ocm_bank][sram_addr]
                                            out_val = np.int32(raw_val)

                                            sram_addr += 1

                                            iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc)
                                            print(f"  > iy_nhwc: {iy_nhwc}")
                                            if not first_pixel_printed and ib == 0:
                                                print(f"\n--- Tracing first raw pixel for Bundle {ib} ---")
                                                print(f"  > Raw val from OCM: {raw_val} at coords (iyh={i_yh}, iyw={i_yw}, iyc={i_yc})")
                                                print(f"  > Stride check: (i_yh - {b['csh_shift']}) % {b['csh']} = {(i_yh - b['csh_shift']) % b['csh']}")
                                                print(f"  > Stride check: (i_yw - {b['csw_shift']}) % {b['csw']} = {(i_yw - b['csw_shift']) % b['csw']}")


                                            # --- ADD P PASSES ---
                                            if b['p'] > 1:
                                                if ip == b['p'] - 1:
                                                    out_val += self.mem['nhwc'][iy_nhwc]
                                                elif ip == 0:
                                                    self.mem['nhwc'][iy_nhwc] = out_val
                                                    continue
                                                else:
                                                    self.mem['nhwc'][iy_nhwc] += out_val
                                                    continue
                                            
                                            # --- CONV STRIDING ---
                                            if (i_yh - b['csh_shift']) % b['csh'] != 0 or \
                                               (i_yw - b['csw_shift']) % b['csw'] != 0:
                                                continue

                                            if not first_pixel_printed and ib == 0:
                                                print(f"  > Passed striding check.")

                                            i_yh = (i_yh - b['csh_shift']) // b['csh']
                                            i_yw = (i_yw - b['csw_shift']) // b['csw']
                                            
                                            # --- ADD BIAS ---
                                            if b.get('is_bias', False):
                                                bias = int(self.mem['b'][i_bias])
                                                out_val = (out_val << b['b_val_shift']) + (bias << b['b_bias_shift'])
                                                if not first_pixel_printed and ib == 0:
                                                    print(f"  > After bias add: {out_val}")
                                                
                                            # --- CORE ACT ---
                                            out_val = self._quant_lrelu(out_val, b['ca_nzero'], b['ca_shift'], b['ca_pl_scale'])

                                            if not first_pixel_printed and ib == 0:
                                                print(f"  > After quant_lrelu (final value for nhwc_buf): {out_val}")
                                                first_pixel_printed = True

                                            # --- RESIDUAL ADD ---
                                            if b.get('add_in_buffer_idx', -1) != -1:
                                                # Need to re-calculate iy_nhwc for the *post-stride* dimensions
                                                add_iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, b['n'], b['ch'], b['cw'], b['co'])
                                                add_val = int(self.mem['add_buffers'][b['add_in_buffer_idx']][add_iy_nhwc])
                                                out_val += add_val
                                                out_val = self._quant_lrelu(out_val, b['aa_nzero'], b['aa_shift'], b['aa_pl_scale'])
                                            
                                            # This is where the C code does tile_write, but that is complex.
                                            # A better approach is to assemble the full NHWC buffer first,
                                            # then do pooling and packing once at the end of the bundle.
                                            final_iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, b['n'], b['ch'], b['cw'], b['co'])
                                            if final_iy_nhwc < nhwc_buf.size:
                                                nhwc_buf[final_iy_nhwc] = out_val
                                
                                # --- Signal Done Reading ---
                                self.mmio.write((self.REG_OFFSETS['A_DONE_READ'] + ocm_bank) * 4, 1)

            # --- Post-Bundle Processing (Pooling, Packing) ---
            print(f"  > Post-processing bundle {ib}...")
            
            # --- DEBUG: Print the calculated NHWC buffer before pooling/packing ---
            print(f"  > Calculated NHWC buffer (first 16 values): {nhwc_buf[:16]}")
            # --- End DEBUG Print ---

            self._perform_pooling_and_packing(nhwc_buf, p_out_buffer, b)
            
            # --- Signal Bundle Done ---
            self.mmio.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)

            # if ib == 0:
            #     print("\nDEBUG: Stopping after bundle 0 for inspection.")
            #     break

        end_time = time.time()
        print(f"--- Model Inference Finished in {end_time - start_time:.4f} seconds ---")
        
        # Reset accelerator start signal now that inference is complete
        # self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 0)
        
        return self._get_final_output()

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
        Implements the exact C runtime shift_round behavior using numpy,
        which correctly handles rounding half to the nearest even number.
        shift_round(n, s) === np.around(n / 2**s)
        """
        if s <= 0:
            return np.int32(n >> s if s < 0 else n)
        
        # Using np.around correctly mimics the C macro's tie-breaking behavior
        # and the developer's own comment in runtime.h
        return np.int32(np.around(n / (2**s)))

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

            # --- DEBUG: Print the data after pooling is complete ---
            print(f"    > Post-pooling data shape: {processed_data.shape}")
            print(f"    > Post-pooling data (first 16 flat values): {processed_data.flatten()[:16]}")
            # --- End DEBUG Print ---

        # --- Flatten and Pack ---
        output_words = processed_data.flatten()
        
        # --- DEBUG: Print the final flattened words before packing ---
        print(f"    > Pre-packing data (first 16 values): {output_words[:16]}")
        # --- End DEBUG Print ---
        
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

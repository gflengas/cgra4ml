import pynq
import numpy as np
import json
import os

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
        
        print(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.defines = config['defines']
        self.bundles = config['bundles']
        self.mem = {}
        
        self._str_to_dtype = {
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
            'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
            'float32': np.float32, 'float64': np.float64
        }
        
        # Correct, minimal set of register offsets from runtime.h
        self.REG_OFFSETS = {
            'A_START': 0x0, 'A_DONE_READ': 0x1, 'A_DONE_WRITE': 0x3, 'A_OCM_BASE': 0x5,
            'A_PARAMS_BASE': 0x7, # This was called A_WEIGHTS_BASE in C, renaming for clarity
            'A_BUNDLE_DONE': 0x8, 'A_N_BUNDLES_1': 0x9,
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
        
        # Create the 'parameters' buffer that holds all bundle configs
        # Each bundle has 8 x 32-bit parameters
        self.mem['params'] = pynq.allocate(shape=(defs['N_BUNDLES'], 8), dtype=np.uint32)

        print("Memory allocation complete.")

    def model_setup(self, wbx_path: str):
        print(f"\nSetting up model from {wbx_path}...")
        
        # 1. Get buffer sizes directly from the allocated memory buffers.
        # This is the correct, robust way to do this.
        w_bytes = self.mem['w'].nbytes
        b_bytes = self.mem['b'].nbytes
        
        # Load the wbx.bin file
        with open(wbx_path, 'rb') as f:
            wbx_data = f.read()

        # 2. Slice the data and copy it into the PYNQ buffers
        np.copyto(self.mem['w'], np.frombuffer(wbx_data[:w_bytes], dtype=self.mem['w'].dtype))
        np.copyto(self.mem['b'], np.frombuffer(wbx_data[w_bytes : w_bytes + b_bytes], dtype=self.mem['b'].dtype))
        np.copyto(self.mem['x'], np.frombuffer(wbx_data[w_bytes + b_bytes:], dtype=self.mem['x'].dtype))
        
        self.mem['w'].flush()
        self.mem['b'].flush()
        self.mem['x'].flush()
        print("Data copy complete.")

        # 2. Pre-load all bundle parameters into the 'params' buffer
        print("Pre-loading all bundle parameters...")
        params_buf = self.mem['params']
        for ib, b in enumerate(self.bundles):
            x_buf = self.mem['x'] if b['in_buffer_idx'] == -1 else self.mem['out_buffers'][b['in_buffer_idx']]
            o_buf = self.mem['y'] if b['out_buffer_idx'] == -1 else self.mem['out_buffers'][b['out_buffer_idx']]
            
            params_buf[ib][0] = self.mem['w'].physical_address
            params_buf[ib][1] = x_buf.physical_address
            params_buf[ib][2] = o_buf.physical_address
            params_buf[ib][3] = self.mem['b'].physical_address
            params_buf[ib][4] = b['b_offset']
            params_buf[ib][5] = 0 # This seems to be an unused 'w_offset' in the C code
            
            header = b['header']
            params_buf[ib][6] = header & 0xFFFFFFFF
            params_buf[ib][7] = header >> 32
        
        params_buf.flush()
        print("Parameter loading complete.")

        # 3. Configure the accelerator with the base addresses
        self.mmio.write(self.REG_OFFSETS['A_N_BUNDLES_1'] * 4, self.defines['N_BUNDLES'] - 1)
        self.mmio.write(self.REG_OFFSETS['A_PARAMS_BASE'] * 4, self.mem['params'].physical_address)
        self.mmio.write(self.REG_OFFSETS['A_OCM_BASE'] * 4, self.mem['ocm'].physical_address)
        print("Register configuration complete.")
        print("Model setup finished.")


    def model_run(self, input_data=None):
        print("\n--- Starting Model Run ---")

        if input_data is not None:
            np.copyto(self.mem['x'], np.frombuffer(input_data, dtype=np.int8))
            self.mem['x'].flush()

        config_base = self.mmio

        for ib, b in enumerate(self.bundles):
            print(f"--- Processing Bundle {ib} ---")
            
            # Tell the hardware which bundle to run. This is done ONCE per bundle.
            config_base.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, ib)

            # This is also crucial: clear DONE_WRITE before starting a new bundle
            config_base.write(self.REG_OFFSETS['A_DONE_WRITE'] * 4, 0) 

            for l in range(b['l']): # Loop over tiles
                for p in range(b['p']): # Loop over passes within a tile
                    print(f"  Tile {l}, Pass {p}")

                    # 1. Start the accelerator for this pass
                    config_base.write(self.REG_OFFSETS['A_START'] * 4, 1)
                    # No need to wait for A_DONE_READ explicitly if hardware handles it automatically after A_START
                    # However, leaving it in for now if it helps ensure the HW picks up the config.
                    # In a simplified hardware, A_DONE_READ might not even be used after A_START.
                    while config_base.read(self.REG_OFFSETS['A_DONE_READ'] * 4) == 0:
                        pass
                    config_base.write(self.REG_OFFSETS['A_START'] * 4, 0)

                    # 2. Wait for the accelerator to be done writing THIS SPECIFIC TILE to OCM
                    # This now waits for the simple 1/0 flag.
                    while config_base.read(self.REG_OFFSETS['A_DONE_WRITE'] * 4) == 0:
                        pass
                    
                    # 3. Process the tile on CPU
                    print(f"    Processing tile {l} on CPU for pass {p}...")
                    y_tile_ocm = self.mem['ocm'][l % 2]
                    o_buf = self.mem['y'] if b['out_buffer_idx'] == -1 else self.mem['out_buffers'][b['out_buffer_idx']]
                    # CORRECTED: Removed the extra config_base argument from the call
                    self.process_tile_py(y_tile_ocm, o_buf, b, l, 0, p)

                    # 4. CRITICAL: Clear A_DONE_WRITE to signal the hardware that the CPU has consumed the tile
                    config_base.write(self.REG_OFFSETS['A_DONE_WRITE'] * 4, 0)


        print("\n--- Model Run Finished ---")
        last_bundle = self.bundles[-1]
        if last_bundle['is_softmax']:
            output_sum = np.sum(self.mem['y'])
            if output_sum != 0:
                self.mem['y'][:] = self.mem['y'] / output_sum
        return self.mem['y']

    def process_tile_py(self, y_tile_in, o_buf, b, il, n, p):
        pe_rows = self.defines['PE_ROWS']
        pe_cols = self.defines['PE_COLS']
        y_tile_full = y_tile_in.copy().reshape(pe_rows, pe_cols)
        y_tile = y_tile_full[:, :b['coe']]

        if b['is_bias'] and p == (b['p'] - 1): # p is always 0, so this check is simpler
            bias_vals = self.mem['b'][b['b_offset'] : b['b_offset'] + b['coe']]
            bias_shifted = (bias_vals.astype(np.int64) << b['b_val_shift']) >> b['b_bias_shift']
            y_tile += bias_shifted.astype(y_tile.dtype)

        if b['ca_nzero']:
            y_tile = (y_tile.astype(np.int64) * b['ca_pl_scale']) >> b['ca_shift']
            y_tile = y_tile.astype(self.mem['ocm'].dtype)
        
        y_post_act = y_tile
        if b['add_in_buffer_idx'] != -1 and p == (b['p'] - 1):
             pass
        if b['aa_nzero'] and p == (b['p'] - 1):
            y_post_act = (y_post_act.astype(np.int64) * b['aa_pl_scale']) >> b['aa_shift']
            y_post_act = y_post_act.astype(self.mem['ocm'].dtype)

        if b['pool'] != 'POOL_NONE' and b['pa_nzero']:
            y_post_act = (y_post_act.astype(np.int64) * b['pa_pl_scale']) >> b['pa_shift']
            y_post_act = y_post_act.astype(self.mem['ocm'].dtype)

        nhwc_buf = self.mem['nhwc']
        for r in range(pe_rows):
            oh = il * b['ch'] + r
            if oh >= b['oh']: continue
            current_row_data = y_post_act[r, :]
            
            if b['pool'] != 'POOL_NONE':
                nhwc_idx = n * b['h'] * b['oc'] + oh * b['oc']
                if (oh % b['psh']) == 0:
                    nhwc_buf[nhwc_idx : nhwc_idx + b['oc']] = current_row_data
                else:
                    if b['pool'] == 'POOL_MAX':
                        current_max = nhwc_buf[nhwc_idx : nhwc_idx + b['oc']]
                        nhwc_buf[nhwc_idx : nhwc_idx + b['oc']] = np.maximum(current_max, current_row_data)
                    elif b['pool'] == 'POOL_AVG':
                        nhwc_buf[nhwc_idx : nhwc_idx + b['oc']] += current_row_data

            should_save, y_to_save, oh_p = False, None, oh
            if b['pool'] != 'POOL_NONE':
                if (oh + 1) % b['psh'] == 0:
                    should_save, oh_p = True, (oh + 1) // b['psh'] - 1
                    nhwc_idx = n * b['h'] * b['oc'] + oh * b['oc']
                    y_to_save = nhwc_buf[nhwc_idx : nhwc_idx + b['oc']].copy()
                    if b['pool'] == 'POOL_AVG': y_to_save = y_to_save // b['pkh']
            else:
                should_save, y_to_save = True, current_row_data

            if should_save:
                if b['is_flatten'] or b['is_softmax']:
                    # This logic needs to be ported similarly if these layers can have pooling
                    pass 
                else:
                    x_bits = self.defines['X_BITS_L2']
                    if x_bits != 0:
                        for c, val in enumerate(y_to_save):
                            o_idx = n * b['oh'] * b['ow'] * b['oc'] + oh_p * b['ow'] * b['oc'] + c
                            bit_idx, bit_offset = o_idx * x_bits, (o_idx * x_bits) % 8
                            byte_idx = bit_idx // 8
                            mask = (1 << x_bits) - 1
                            o_buf[byte_idx] &= ~(mask << bit_offset)
                            o_buf[byte_idx] |= (val & mask) << bit_offset
                            if x_bits + bit_offset > 8:
                                o_buf[byte_idx+1] &= ~(mask >> (8 - bit_offset))
                                o_buf[byte_idx+1] |= (val & mask) >> (8-bit_offset)
                    else:
                        o_buf_view = o_buf.view(y_to_save.dtype)
                        o_idx = n * b['oh'] * b['ow'] * b['oc'] + oh_p * b['ow'] * b['oc']
                        o_buf_view[o_idx : o_idx + y_to_save.size] = y_to_save
        
        # --- NO LONGER NEEDED HERE: The A_DONE_WRITE clear is now done in model_run per tile. ---
        # if (il + 1) == b['l']:
        #     print(f"    Last tile of bundle, clearing A_DONE_WRITE to signal completion.")
        #     config_base.write(self.REG_OFFSETS['A_DONE_WRITE'] * 4, 0)



    def __del__(self):
        print("\nReleasing memory buffers.")
        for name, buf in self.mem.items():
            if hasattr(buf, 'freebuffer'):
                buf.freebuffer()
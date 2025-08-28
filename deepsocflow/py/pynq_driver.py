import pynq
import numpy as np
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
        print("Register configuration complete.")
        print("Model setup finished.")

    def model_run(self, input_data=None):
        print("\n--- Starting Model Run ---")

        if input_data is not None:
            np.copyto(self.mem['x'], np.frombuffer(input_data, dtype=np.int8))
            self.mem['x'].flush()

        config_base = self.mmio
        ocm_bank = 1 # Will be flipped to 0 on first iteration

        for ib, b in enumerate(self.bundles):
            print(f"\n--- Processing Bundle {ib} ---")
            print(f"  Bundle Parameters: {b}")
            
            is_last_bundle = (ib == len(self.bundles) - 1)
            o_buf = self.mem['y'] if is_last_bundle else self.mem['out_buffers'][b['out_buffer_idx']]

            # Create a temporary buffer to reassemble the full output image for this layer
            nhwc_buf = np.zeros(b['n'] * b['h'] * b['oc'], dtype=np.int32)

            # This loop structure implements the OCM double-buffering handshake
            # It is a simplified version of the more complex C-runtime loop.
            for l in range(b['l']):
                for p in range(b['p']):
                    print(f"  Tile {l}, Pass {p}")

                    ocm_bank = 1 - ocm_bank # Flip bank for double buffering (0->1 or 1->0)

                    # Start the accelerator
                    config_base.write(self.REG_OFFSETS['A_START'] * 4, 1)
                    
                    # Wait for the accelerator to finish writing to the current OCM bank
                    while config_base.read((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4) == 0:
                        pass
                    
                    config_base.write(self.REG_OFFSETS['A_START'] * 4, 0)
                    
                    # Invalidate the cache for the OCM buffer so we get the new data from the PL
                    y_tile_ocm = self.mem['ocm'][ocm_bank]
                    y_tile_ocm.invalidate()
                    
                    print(f"    Processing tile {l} on CPU for pass {p}...")
                    self.process_tile_py(y_tile_ocm, nhwc_buf, b, l, 0, p)
                    
                    # Reset the hardware's DONE_WRITE flag
                    config_base.write((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4, 0)
                    # Signal to hardware that the CPU is done reading from this OCM bank
                    config_base.write((self.REG_OFFSETS['A_DONE_READ'] + ocm_bank) * 4, 1)

            # After all passes and tiles, perform pooling/packing on the assembled buffer
            self._perform_pooling_and_packing(nhwc_buf, o_buf, b)
            
            # Signal that the entire bundle is done
            config_base.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)

        print("\n--- Model Run Finished ---")
        return self._get_final_output()

    def process_tile_py(self, y_tile_in, nhwc_buf, b, il, n, p):
        """
        Processes a single tile of output data from the OCM.
        This function's main job is to apply bias and activation, then place
        the resulting data into the correct location in the larger NHWC buffer.
        It does NOT perform pooling.
        """
        # --- DEBUG PRINT: Verify raw OCM input to CPU ---
        print(f"    DEBUG: Raw OCM data entering process_tile_py (B:{b['ib']}, L:{il}, P:{p}): {y_tile_in[:32]}")
        
        pe_rows = self.defines['PE_ROWS']
        pe_cols = self.defines['PE_COLS']
        x_bits_l2 = self.defines['X_BITS_L2']
        x_bits = 1 << x_bits_l2

        y_tile = y_tile_in.copy().reshape(pe_rows, pe_cols)

        # Determine the number of valid channels to process for this specific pass
        p_offset = p * self.defines['PE_COLS']
        valid_channels_in_pass = min(b['coe'], b['oc'] - p_offset)
        
        # --- Per-Pixel Processing (mimicking the C-runtime loop) ---
        for r_idx in range(pe_rows):
            oh = il * pe_rows + r_idx
            if oh >= b['h']: continue
            
            for c_idx in range(valid_channels_in_pass):
                oc = p_offset + c_idx
                if oc >= b['oc']: continue
                
                val = y_tile[r_idx, c_idx].astype(np.int64)

                # --- Apply Bias ---
                if b['is_bias'] and p == (b['p'] - 1):
                    bias_val = self.mem['b'][b['b_offset'] + oc]
                    val = (val << b['b_val_shift']) + (bias_val.astype(np.int64) << b['b_bias_shift'])

                # --- Core Activation (ca_) ---
                if b['ca_nzero']:
                    if val >= 0: val = val << b['ca_pl_scale']
                    
                    float_shifted = float(val) / (2**b['ca_shift'])
                    val = np.around(float_shifted).astype(np.int64)
                    
                    min_clip = -(2**(x_bits - b['ca_pl_scale'] - 1))
                    max_clip = (2**(x_bits - 1)) - 1
                    val = np.clip(val, min_clip, max_clip)

                # --- Residual Add ---
                nhwc_idx = n * b['h'] * b['oc'] + oh * b['oc'] + oc
                if b['add_in_buffer_idx'] != -1:
                    val += self.mem['add_buffers'][b['add_in_buffer_idx']][nhwc_idx]

                    # --- Adder Activation (aa_) ---
                    if b['aa_nzero']:
                        if val >= 0: val = val << b['aa_pl_scale']
                        
                        float_shifted = float(val) / (2**b['aa_shift'])
                        val = np.around(float_shifted).astype(np.int64)

                        min_clip = -(2**(x_bits - b['aa_pl_scale'] - 1))
                        max_clip = (2**(x_bits - 1)) - 1
                        val = np.clip(val, min_clip, max_clip)
                
                # Place the final processed value into the correct position
                nhwc_buf[nhwc_idx] = val

    def _perform_pooling_and_packing(self, nhwc_buf, o_buf, b):
        """
        Performs the final pooling, flattening, and packing operations on the
        fully assembled NHWC buffer for a bundle.
        """
        # If the bundle was a flatten layer, there is no 2D structure for pooling.
        # The output is already the flattened data.
        if b.get('is_flatten', False):
            processed_data = nhwc_buf
        else:
            # Reshape the flat NHWC buffer into a 2D image representation (height x channels)
            img = nhwc_buf.reshape(b['h'], b['oc'])

            # --- Perform Pooling on the full image ---
            if b['pool'] != 'POOL_NONE':
                pooled_rows = b['oh']
                pooled_cols = b['ow']
                pooled_img = np.zeros((pooled_rows, pooled_cols), dtype=np.int32)
                
                for r in range(pooled_rows):
                    for c in range(pooled_cols):
                        r_start, c_start = r * b['psh'], c * b['psw']
                        r_end, c_end = r_start + b['pkh'], c_start + b['pkw']
                        window = img[r_start:r_end, c_start:c_end]
                        
                        if b['pool'] == 'POOL_MAX':
                            pooled_img[r, c] = np.max(window)
                        elif b['pool'] == 'POOL_AVG':
                            pooled_img[r, c] = np.mean(window).astype(np.int32)
                
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
            
            # Now apply the softmax exponentiation
            exp_values = np.exp(float_words)
            sum_exp_values = np.sum(exp_values)
            
            if sum_exp_values != 0:
                final_output = exp_values / sum_exp_values
            else:
                final_output = exp_values
        else:
            final_output = final_output_words

        # Return only the valid part of the output buffer
        return final_output[:last_bundle['o_words']]
        
    def __del__(self):
        print("\nReleasing memory buffers.")
        for name, buf in self.mem.items():
            if hasattr(buf, 'freebuffer'):
                buf.freebuffer()

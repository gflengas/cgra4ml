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
        y_type = self._str_to_dtype[defs['Y_TYPE_str']]
        b_type = self._str_to_dtype[defs['B_TYPE_str']]
        o_type = self._str_to_dtype[defs['O_TYPE_str']]

        # --- Fix: Allocate OCM as int32 to match sign-extended DMA writes ---
        print("  > NOTE: Allocating OCM with dtype=np.int32 to match HW DMA behavior.")
        self.mem['ocm'] = pynq.allocate(shape=(2, defs['PE_COLS'] * defs['PE_ROWS']), dtype=np.int32)
        # -----------------------------------------------------------------------
        
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


    def model_run(self, input_data=None, debug=False):
        """
        Executes the model inference on the accelerator, mimicking the C-runtime.
        """
        # --- Fix for non-determinism: Force a complete HW re-initialization ---
        print("  > Forcing clean state: Zeroing SW buffers and re-initializing all HW registers...")
        # 1. Zero-out all intermediate/output software buffers
        self.mem['nhwc'].fill(0)
        self.mem['out_buffers'].fill(0)
        self.mem['y'].fill(0)
        if 'add_buffers' in self.mem:
            self.mem['add_buffers'].fill(0)
        
        # 2. Force a full re-initialization of all hardware control registers
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
        self.mmio.write(self.REG_OFFSETS['A_W_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_X_DONE'] * 4, 0)
        self.mmio.write(self.REG_OFFSETS['A_O_DONE'] * 4, 0)
        # --------------------------------------------------------------------------

        # --- Fix for timing/race condition: Add a small delay for HW state to settle ---
        time.sleep(0.001)  # 1 millisecond delay
        # ------------------------------------------------------------------------------

        if input_data is not None:
            # This assumes the input buffer is self.mem['x'] for the first layer
            np.copyto(self.mem['x'], input_data.flatten())
            self.mem['x'].flush()

        print("\n--- Starting Model Inference ---")
        start_time = time.time()

        # Start the accelerator with a pulse (1 -> 0)
        self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 1)
        self.mmio.write(self.REG_OFFSETS['A_START'] * 4, 0)

        # The one-time status check is no longer needed
        # time.sleep(0.1) 
        # w_done = self.mmio.read(self.REG_OFFSETS['A_W_DONE'] * 4)
        # x_done = self.mmio.read(self.REG_OFFSETS['A_X_DONE'] * 4)
        # o_done = self.mmio.read(self.REG_OFFSETS['A_O_DONE'] * 4)
        # print(f"  > HW Status after start: W_DONE={w_done}, X_DONE={x_done}, O_DONE={o_done}")

        ocm_bank = 1  # Will be flipped to 0 at the start of the first loop

        for ib, b in enumerate(self.bundles):
            print(f"Executing Bundle {ib}/{len(self.bundles)-1}...")
            
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

                                # --- Wait for Accelerator ---
                                while not self.mmio.read((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4):
                                    time.sleep(0.00001) # Small sleep to avoid busy-waiting too aggressively
                                
                                self.mem['ocm'][ocm_bank].invalidate()
                                
                                if iw_kw2 == 0 and it == 0:
                                    print(f"\n--- Reading OCM Bank {ocm_bank} for Bundle {ib} (ip={ip}, it={it}, iw_kw2={iw_kw2}) ---")
                                    # Print first 32 values to verify fix, without cluttering output
                                    print(np.int16(self.mem['ocm'][ocm_bank][:32]))
                                
                                self.mmio.write((self.REG_OFFSETS['A_DONE_WRITE'] + ocm_bank) * 4, 0)
                                
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
                                            
                                            # Fix: Read the 32-bit word, then cast to 16-bit to get the correct value
                                            raw_val = self.mem['ocm'][ocm_bank][sram_addr]
                                            out_val = int(np.int16(raw_val))
                                            sram_addr += 1

                                            iy_nhwc = self._flatten_nhwc(i_yn, i_yh, i_yw, i_yc, yn, yh, yw, yc)

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
                                            
                                            i_yh = (i_yh - b['csh_shift']) // b['csh']
                                            i_yw = (i_yw - b['csw_shift']) // b['csw']
                                            
                                            # --- ADD BIAS ---
                                            if b.get('is_bias', False):
                                                bias = int(self.mem['b'][i_bias])
                                                out_val = (out_val << b['b_val_shift']) + (bias << b['b_bias_shift'])
                                                
                                            # --- CORE ACT ---
                                            out_val = self._quant_lrelu(out_val, b['ca_nzero'], b['ca_shift'], b['ca_pl_scale'])

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
            self._perform_pooling_and_packing(nhwc_buf, p_out_buffer, b)
            
            # --- Signal Bundle Done ---
            self.mmio.write(self.REG_OFFSETS['A_BUNDLE_DONE'] * 4, 1)

            if ib == 0:
                print("\nDEBUG: Stopping after bundle 0 for inspection.")
                break

        end_time = time.time()
        print(f"--- Model Inference Finished in {end_time - start_time:.4f} seconds ---")
        
        # Reset accelerator start signal - NO LONGER NEEDED as it's now a pulse at the beginning.
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

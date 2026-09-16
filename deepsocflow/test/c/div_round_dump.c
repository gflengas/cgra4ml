/* Dumps runtime.h's div_round macro over a sweep, so the Python side can be
 * pinned against the C itself rather than against a re-reading of it.
 *
 * The macro's tie-break is not the obvious one, and average pooling's whole
 * bit-exactness claim rests on reproducing it - so it is transcribed here
 * verbatim from deepsocflow/c/runtime.h and compared, rather than
 * reimplemented. Kept minimal on purpose: including runtime.h itself would drag
 * in config_fw.h and the whole Memory_st/DPI surface.
 *
 *   cc -o div_round_dump div_round_dump.c && ./div_round_dump
 *
 * Prints "a b result" per line. Consumed by
 * deepsocflow/test/py/test_brevitas_conv.py::test_div_round_matches_c.
 */
#include <stdio.h>
#include <stdint.h>

typedef int32_t i32;

/* verbatim from deepsocflow/c/runtime.h */
#define div_round(a, b) (((a)+((b)/2) - (~((b)|(a)/(b)) &1))/(b))

int main(void) {
    /* Window sizes an 'valid' pool can produce (PKH*PKW), and a sum range wide
     * enough to cover count * the full signed 8-bit activation range. */
    const i32 counts[] = {1, 2, 3, 4, 6, 9, 12, 16, 25, 36};
    const int n_counts = (int)(sizeof(counts) / sizeof(counts[0]));

    for (int ci = 0; ci < n_counts; ci++) {
        i32 b = counts[ci];
        i32 lo = -128 * b, hi = 127 * b;
        for (i32 a = lo; a <= hi; a++)
            printf("%d %d %d\n", a, b, (i32)div_round(a, b));
    }
    return 0;
}

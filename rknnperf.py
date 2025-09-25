#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
from rknn.api import RKNN

def evaluate_performance(model_path, verbose=False, device_id=None, multi_core=False):
    """
    Evaluate the performance of an RKNN model.

    Parameters:
    - model_path: Path to the RKNN model file.
    - verbose: Boolean flag to enable detailed performance logging.
    - multi_core: Boolean flag to enable multi-core evaluation.
    """
    try:
        rknn = RKNN(verbose=verbose)
        print(f"Loading model from {model_path}")
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        # Assuming the target is 'rk3588' for this example. Adjust accordingly.
        ret = rknn.init_runtime(target='rk3588', device_id=device_id, core_mask=RKNN.NPU_CORE_0_1_2 if multi_core else RKNN.NPU_CORE_0, perf_debug=verbose)
        if ret != 0:
            print('Init runtime environment failed!')
            sys.exit(ret)

        print('--> Evaluating performance')
        rknn.eval_perf()

    except Exception as e:
        print("An error occurred during the evaluation:")
        print(e)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the performance of an RKNN model.")
    parser.add_argument("model", type=str, help="Path to the RKNN model file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed performance logging.")
    parser.add_argument("-s", "--device_id", type=str, help="Device ID.")
    parser.add_argument("-c", "--multi_core", action="store_true", help="Enable multi-core evaluation.")

    args = parser.parse_args()

    evaluate_performance(args.model, args.verbose, args.device_id, args.multi_core)

if __name__ == "__main__":
    main()
